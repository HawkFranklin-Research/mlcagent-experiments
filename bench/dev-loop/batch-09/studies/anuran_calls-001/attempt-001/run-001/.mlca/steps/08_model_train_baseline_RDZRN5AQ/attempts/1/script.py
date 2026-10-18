import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

protocol = json.loads(inputs['_protocol'].read_text())

def get_protocol(key, required=True):
    if key not in protocol:
        if required:
            raise KeyError(f"Missing protocol key '{key}'. Available: {list(protocol.keys())}")
        return None
    return protocol[key]

targets = get_protocol('targets')
excluded = get_protocol('excluded_columns')
group_col = get_protocol('group_column', required=False)
task = get_protocol('task')

train = pd.read_parquet(inputs['train'])

feature_cols = [c for c in train.columns if c not in targets]
if excluded:
    feature_cols = [c for c in feature_cols if c not in excluded]
if group_col and group_col in feature_cols:
    feature_cols.remove(group_col)
for c in list(feature_cols):
    s = train[c]
    if s.dtype == object or str(s.dtype) in ('string', 'category'):
        if s.nunique(dropna=False) == len(train):
            feature_cols.remove(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique() == len(train) and any(k in c.lower() for k in ('id', 'key', 'code', 'index')):
            feature_cols.remove(c)

with open(inputs['06_preprocess_fit_preprocessing_DB5T0R19/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['07_features_retain_features_05E7EJVW/selector_all.pkl'], 'rb') as f:
    sel = pickle.load(f)

def includes_preprocessing(obj):
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            if isinstance(step, ColumnTransformer):
                return True
            if isinstance(step, Pipeline) and includes_preprocessing(step):
                return True
    return False

steps = []
if includes_preprocessing(sel):
    steps.append(('sel', sel))
else:
    steps.append(('pre', pre))
    steps.append(('sel', sel))
steps.append(('clf', MultiOutputClassifier(LogisticRegression(max_iter=2000))))
pipeline = Pipeline(steps)

expected = None
for obj in (pre, sel):
    if hasattr(obj, 'feature_names_in_'):
        expected = list(obj.feature_names_in_)
        break
if expected is not None:
    forbidden = set(targets) | set(excluded or []) | ({group_col} if group_col else set())
    if any(c in forbidden for c in expected):
        raise ValueError("Saved pipeline feature_names_in_ contains target/excluded/group columns")
    feature_cols = expected
feature_cols = [c for c in feature_cols if c in train.columns]

X_train = train[feature_cols]
y_train = train[targets]

mlca_runtime.fit_record(pipeline, X_train, 'train', y_train)
mlca_runtime.save_model(pipeline, 'model.pkl')
print(f"features={len(feature_cols)} rows={len(train)} targets={len(targets)}")
