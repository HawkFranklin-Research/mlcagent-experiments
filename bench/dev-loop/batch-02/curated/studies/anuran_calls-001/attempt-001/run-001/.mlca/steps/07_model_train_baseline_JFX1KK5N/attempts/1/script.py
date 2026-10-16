import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

required = ['targets', 'excluded_columns', 'group_column', 'task']
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing keys {missing} in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol['group_column']
task = protocol['task']

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_Z1YWJ829/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_retain_features_M3TS3EMK/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)

def is_key_like(series):
    if pd.api.types.is_float_dtype(series):
        return False
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return series.nunique(dropna=False) == len(series)
    return False

def get_feature_names(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    return None

selector_includes_pre = False
if isinstance(selector, Pipeline):
    first_step = list(selector.named_steps.values())[0]
    if isinstance(first_step, (ColumnTransformer, Pipeline)):
        selector_includes_pre = True

if selector_includes_pre:
    raw_cols = get_feature_names(selector) or get_feature_names(preprocessor)
else:
    raw_cols = get_feature_names(preprocessor) or get_feature_names(selector)

if raw_cols is None:
    raw_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
    raw_cols = [c for c in raw_cols if not is_key_like(train[c])]

for c in raw_cols:
    if c not in train.columns:
        raise ValueError(f"Feature column {c} not in training data")
    if c in targets or c in excluded or c == group:
        raise ValueError(f"Feature column {c} is a target/excluded/group column")

X = train[raw_cols].copy()
y = train[targets].copy()

clf = MultiOutputClassifier(LogisticRegression(max_iter=2000))

if selector_includes_pre:
    final = Pipeline([
        ('selector', selector),
        ('clf', clf)
    ])
else:
    final = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('clf', clf)
    ])

mlca_runtime.fit_record(final, X, 'train', y=y)
mlca_runtime.save_model(final, 'model.pkl')
