import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing {key!r}; available keys: {sorted(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_KGJNVG5D/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

def is_key_like(s):
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        return s.nunique(dropna=False) == len(s) and len(s) > 1
    if pd.api.types.is_integer_dtype(s.dtype) and s.notna().all() and s.is_unique:
        v = np.sort(s.to_numpy())
        return len(v) > 1 and np.array_equal(v, np.arange(v[0], v[0] + len(v)))
    return False

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

candidate = [c for c in train.columns if c not in drop and not is_key_like(train[c])]
expected = list(getattr(preprocessor, "feature_names_in_", []))
if expected:
    bad = set(expected) & drop
    if bad:
        raise ValueError(f"Preprocessor expects forbidden columns: {sorted(bad)}")
    missing = [c for c in expected if c not in candidate]
    if missing:
        raise ValueError(f"Preprocessor expects undeclared columns: {missing}")
    feature_cols = [c for c in expected if c in candidate]
else:
    feature_cols = candidate

if not feature_cols:
    raise ValueError("No feature columns remain")

try:
    n_features = len(preprocessor.get_feature_names_out())
except Exception:
    n_features = preprocessor.transform(train[feature_cols].head(5)).shape[1]

k = min(20, n_features)
if k < 1:
    raise ValueError("No preprocessed features available for selection")

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("select", SelectKBest(score_func=mutual_info_classif, k=k)),
])
X = train[feature_cols]
y = train[targets[0]]
pipeline = mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(pipeline, "selector.pkl")

selected = int(pipeline.named_steps["select"].get_support().sum())
mlca_runtime.record_metric("n_preprocessed_features", float(n_features), "train")
mlca_runtime.record_metric("n_selected_features", float(selected), "train")
print("train shape:", train.shape)
print("features:", len(feature_cols), "preprocessed:", n_features, "k:", k)
print("selected:", selected)
