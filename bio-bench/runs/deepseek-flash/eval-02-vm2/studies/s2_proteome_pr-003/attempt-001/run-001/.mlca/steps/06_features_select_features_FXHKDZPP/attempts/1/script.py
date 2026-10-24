import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())


def need(d, key):
    if key not in d:
        raise KeyError(f"Missing {key!r} in _protocol; available top-level keys: {sorted(d.keys())}")
    return d[key]


targets = need(protocol, "targets")
excluded = set(need(protocol, "excluded_columns"))
group_col = need(protocol, "group_column")
task = need(protocol, "task")
if "primary_metric" not in task:
    raise KeyError(f"Missing 'primary_metric' in task; available task keys: {sorted(task.keys())}")

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_YCEBPQHT/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

target = targets[0]
if target not in train.columns:
    raise KeyError(f"Target {target!r} not in train columns")
y = train[target]


def is_key_like(s, col):
    n = len(s)
    if n == 0 or s.nunique(dropna=False) != n:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        name = str(col).lower()
        return any(tok in name for tok in ("id", "code", "key", "sample", "patient", "case", "subject"))
    return False


raw_cols = list(train.columns)
cand = [c for c in raw_cols if c not in set(targets) and c not in excluded and c != group_col]
computed = [c for c in cand if not is_key_like(train[c], c)]

if hasattr(pre, "feature_names_in_"):
    pre_cols = [str(c) for c in pre.feature_names_in_]
    bad = set(pre_cols) & (set(targets) | excluded | {group_col})
    if bad:
        raise ValueError(f"Preprocessor expects excluded/target columns: {sorted(bad)}")
    missing = [c for c in pre_cols if c not in train.columns]
    if missing:
        raise ValueError(f"Preprocessor expects columns missing from train: {missing}")
    feature_cols = pre_cols
else:
    feature_cols = computed

if not feature_cols:
    raise ValueError("No feature columns remain")

front = pre
if isinstance(pre, Pipeline) and any(isinstance(st, SelectKBest) for st in pre.named_steps.values()):
    kept = [(n, st) for n, st in pre.named_steps.items() if not isinstance(st, SelectKBest)]
    front = Pipeline(kept) if kept else "passthrough"


def out_width(obj, X):
    if obj == "passthrough":
        return X.shape[1]
    try:
        return len(obj.get_feature_names_out())
    except Exception:
        try:
            return obj.transform(X).shape[1]
        except Exception:
            return X.shape[1]


X = train[feature_cols].copy()
n_features = out_width(front, X)
k = min(20, n_features)
if k < 1:
    raise ValueError("No features available after preprocessing")

full = Pipeline([
    ("preprocessor", front),
    ("selector", SelectKBest(score_func=mutual_info_classif, k=k)),
])
mlca_runtime.fit_record(full, X, "train", y)

n_selected = int(full.named_steps["selector"].get_support().sum())
mlca_runtime.record_metric("n_features_after_preprocessing", float(n_features), "train")
mlca_runtime.record_metric("n_features_selected", float(n_selected), "train")
mlca_runtime.save_model(full, "selector")
print(f"features_in={len(feature_cols)} features_pre={n_features}")
print(f"k={k} selected={n_selected}")
