import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def get_protocol(key):
    if key not in protocol:
        raise KeyError(f"Missing {key} in _protocol. Available keys: {list(protocol.keys())}")
    return protocol[key]

targets = get_protocol("targets")
excluded = get_protocol("excluded_columns")
group = protocol.get("group_column")
task = get_protocol("task")

train = pd.read_parquet(inputs["train"])
pre = pickle.loads(inputs["05_preprocess_fit_preprocessing_ZR2ARXZ1/preprocessor.pkl"].read_bytes())
sel = pickle.loads(inputs["06_features_retain_features_H6GS4FCS/selector_all.pkl"].read_bytes())

if hasattr(pre, 'feature_names_in_'):
    feature_cols = list(pre.feature_names_in_)
else:
    all_cols = list(train.columns)
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    feature_cols = []
    for c in all_cols:
        if c in drop:
            continue
        s = train[c]
        if s.nunique(dropna=False) == len(train) and (s.dtype == object or str(s.dtype).startswith('string') or pd.api.types.is_integer_dtype(s.dtype)):
            continue
        feature_cols.append(c)

drop_set = set(targets) | set(excluded)
if group:
    drop_set.add(group)
feature_cols = [c for c in feature_cols if c not in drop_set]

X = train[feature_cols]
y = train[targets]

def is_complete_frontend(obj, raw_cols):
    if not isinstance(obj, Pipeline):
        return False
    if hasattr(obj, 'feature_names_in_') and set(obj.feature_names_in_) == set(raw_cols):
        return True
    for _, step in obj.named_steps.items():
        if isinstance(step, ColumnTransformer):
            return True
    return False

if is_complete_frontend(sel, feature_cols):
    frontend = sel
else:
    frontend = Pipeline([('pre', pre), ('sel', sel)])

model = RandomForestRegressor(n_estimators=100, random_state=42)
pipe = Pipeline([('frontend', frontend), ('model', model)])

fitted = mlca_runtime.fit_record(pipe, X, 'train', y=y)
mlca_runtime.save_model(fitted, "alternative_model.pkl")
print(f"train shape: {X.shape}, targets: {targets}")
print(f"features: {len(feature_cols)}")
print("saved alternative_model.pkl")
