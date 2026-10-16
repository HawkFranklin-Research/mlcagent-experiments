import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

def get_protocol_key(key):
    if key not in protocol:
        raise KeyError(f"Missing {key}. Available top-level keys: {list(protocol.keys())}")
    return protocol[key]

targets = get_protocol_key('targets')
excluded = get_protocol_key('excluded_columns')
group_col = get_protocol_key('group_column')
task = get_protocol_key('task')

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_WXDY6VPK/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_retain_features_C7QR720T/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)

feature_cols = None
if isinstance(selector, Pipeline) and hasattr(selector, 'feature_names_in_'):
    feature_cols = list(selector.feature_names_in_)
if feature_cols is None and hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = list(preprocessor.feature_names_in_)
if feature_cols is None:
    all_cols = list(train.columns)
    candidate = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]
    feature_cols = []
    for c in candidate:
        s = train[c]
        if s.nunique(dropna=False) == len(train) and (
            pd.api.types.is_object_dtype(s) or
            pd.api.types.is_string_dtype(s) or
            pd.api.types.is_categorical_dtype(s) or
            pd.api.types.is_integer_dtype(s)
        ):
            continue
        feature_cols.append(c)

for t in targets:
    if t in feature_cols:
        raise ValueError(f"Target {t} found in feature columns")

X_train = train[feature_cols].copy()
y_train = train[targets].copy()

model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))

selector_includes_preprocessor = False
if isinstance(selector, Pipeline):
    if hasattr(selector, 'feature_names_in_') and hasattr(preprocessor, 'feature_names_in_'):
        if list(selector.feature_names_in_) == list(preprocessor.feature_names_in_):
            selector_includes_preprocessor = True
    if not selector_includes_preprocessor:
        first_step = list(selector.named_steps.values())[0]
        if isinstance(first_step, (ColumnTransformer, Pipeline)):
            selector_includes_preprocessor = True

if selector_includes_preprocessor:
    pipeline = Pipeline([
        ('selector', selector),
        ('model', model)
    ])
else:
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', model)
    ])

mlca_runtime.fit_record(pipeline, X_train, 'train', y=y_train)
mlca_runtime.save_model(pipeline, "alternative_model.pkl")

print(f"Train rows: {len(train)}")
print(f"Features: {len(feature_cols)}")
print(f"Targets: {len(targets)}")
print("Saved alternative_model.pkl")
