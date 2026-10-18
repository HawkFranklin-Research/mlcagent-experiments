import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing {key!r}; available top-level keys: {sorted(d.keys())}")
    return d[key]

def is_key_like(s):
    nonnull = s.dropna()
    if len(nonnull) <= 1 or nonnull.nunique() != len(nonnull):
        return False
    if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return True
    return False

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group = need(protocol, "group_column")
task = need(protocol, "task")
json.loads(inputs["05_preprocess_fit_preprocessing_9HEDMWQR/manifest.json"].read_text())
train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_9HEDMWQR/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)
feature_cols = [c for c in train.columns if c not in drop]
feature_cols = [c for c in feature_cols if not is_key_like(train[c])]

if hasattr(pre, "feature_names_in_"):
    pre_cols = list(pre.feature_names_in_)
    if any(c in drop for c in pre_cols):
        raise ValueError("Loaded preprocessor expects target/excluded/group columns")
    if set(pre_cols) == set(feature_cols):
        feature_cols = pre_cols
    else:
        raise ValueError(f"Feature schema mismatch: expected {pre_cols}, got {feature_cols}")

X = train[feature_cols].copy()
y = train[targets].copy()

if isinstance(pre, Pipeline):
    steps = list(pre.steps) + [("selector", "passthrough")]
else:
    steps = [("preprocessor", pre), ("selector", "passthrough")]
pipeline = Pipeline(steps)
fitted = mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(fitted, "selector_all.pkl")
print(f"train rows: {len(train)}, features: {len(feature_cols)}, targets: {len(targets)}")
print(f"steps: {[name for name, _ in fitted.steps]}")
