import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
IN = mlca_runtime.inputs()

train = pd.read_parquet(IN["train"])
with open(IN["05_preprocess_fit_preprocessing_6GF5BX9F/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
proto = json.loads(IN["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"{k} missing; available top-level keys: {sorted(d)}")
    return d[k]

targets = req(proto, "targets")
excluded = req(proto, "excluded_columns") or []
task = req(proto, "task")
group = proto.get("group_column", None)

if isinstance(targets, str):
    targets = [targets]
if isinstance(excluded, str):
    excluded = [excluded]
if len(targets) != 1:
    raise ValueError(f"expected one target, got {targets}")
target_col = targets[0]

forbidden = set(targets) | set(excluded)
if group is not None:
    forbidden.add(group)

def is_key_like(s):
    if s.dtype == object or str(s.dtype).startswith("string") or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == len(s) and len(s) > 1
    if pd.api.types.is_integer_dtype(s.dtype):
        name = str(s.name).lower()
        hint = any(t in name for t in ["id", "code", "key", "index", "seq"])
        return hint and s.nunique(dropna=False) == len(s) and len(s) > 1
    return False

candidates = [c for c in train.columns if c not in forbidden]
feature_cols = [c for c in candidates if not is_key_like(train[c])]

if hasattr(pre, "feature_names_in_"):
    fin = list(pre.feature_names_in_)
    if set(fin) & forbidden:
        raise ValueError("preprocessor feature_names_in_ includes a forbidden column")
    if not set(fin) <= set(train.columns):
        raise ValueError("preprocessor expects columns missing from train")
    feature_cols = fin

try:
    n_transformed = len(pre.get_feature_names_out())
except Exception:
    n_transformed = pre.transform(train[feature_cols].iloc[:1]).shape[1]

k = min(20, n_transformed)
selector = SelectKBest(score_func=mutual_info_classif, k=k)
pipe = Pipeline([("preprocessor", pre), ("select", selector)])

if n_transformed <= 300:
    for obj in (pre, pipe):
        if hasattr(obj, "set_output"):
            try:
                obj.set_output(transform="pandas")
            except Exception:
                pass

X = train[feature_cols]
y = train[target_col]
fitted = mlca_runtime.fit_record(pipe, X, partition="train", y=y)
mlca_runtime.save_model(fitted, "selector.pkl")

mlca_runtime.record_metric("n_features_in", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_features_selected", float(k), "train")
print(f"train rows={len(train)} cols={train.shape[1]}")
print(f"features={len(feature_cols)} transformed={n_transformed} k={k}")
