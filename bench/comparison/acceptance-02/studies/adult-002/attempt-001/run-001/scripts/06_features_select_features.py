import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol)}")

targets = protocol["targets"]
excluded = set(protocol["excluded_columns"] or [])
group = protocol["group_column"]
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_WBRZJAT5/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

def is_key_like(s):
    if s.isna().any():
        return False
    n = len(s)
    if s.nunique(dropna=False) != n:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        lo, hi = int(s.min()), int(s.max())
        return lo in (0, 1) and hi - lo + 1 == n
    return False

forbidden = set(targets) | excluded | ({group} if group else set())
if hasattr(pre, "feature_names_in_"):
    feature_cols = list(pre.feature_names_in_)
    bad = [c for c in feature_cols if c not in train.columns or c in forbidden]
    if bad:
        raise ValueError(f"Preprocessor expects forbidden or missing columns: {bad}")
else:
    allowed = [c for c in train.columns if c not in forbidden]
    feature_cols = [c for c in allowed if not is_key_like(train[c])]
if not feature_cols:
    raise ValueError("No feature columns remain")

X_train = train[feature_cols].copy()
y_train = train[target]

# Try to read preprocessing output width without refitting; if it is not fitted, fit via fit_record.
n_features = None
if hasattr(pre, "get_feature_names_out"):
    try:
        n_features = len(pre.get_feature_names_out())
    except Exception:
        n_features = None
if n_features is None:
    mlca_runtime.fit_record(pre, X_train, "train", y=y_train)
    n_features = pre.transform(X_train).shape[1]
if n_features < 1:
    raise ValueError("Preprocessing produced no features")
k = min(20, n_features)

steps = list(pre.steps) if isinstance(pre, Pipeline) else [("preprocess", pre)]
pipeline = Pipeline(steps + [("selectkbest_final", SelectKBest(mutual_info_classif, k=k))])
mlca_runtime.fit_record(pipeline, X_train, "train", y=y_train)
mlca_runtime.save_model(pipeline, "selector.pkl")

mlca_runtime.record_metric("n_features_input", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_features_preprocessed", float(n_features), "train")
mlca_runtime.record_metric("k_selected", float(k), "train")
print(f"train rows={len(train)}, feature cols={len(feature_cols)}")
print(f"preprocessed cols={n_features}, k={k}")
print("saved selector.pkl")
