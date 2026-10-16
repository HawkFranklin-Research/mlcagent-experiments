import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
need = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in need if k not in proto]
if missing:
    raise KeyError(f"Missing {missing}; available top-level keys: {list(proto)}")

targets = proto["targets"]
excluded = proto["excluded_columns"] or []
group = proto["group_column"]
target_cols = [targets] if isinstance(targets, str) else list(targets)
if not target_cols:
    raise ValueError("No target columns in _protocol['targets']")
target = target_cols[0]

train = pd.read_parquet(inputs["train"])
y = train[target]

with open(inputs["05_preprocess_fit_preprocessing_Y4SGGFB6/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

drop = set(target_cols) | set(excluded) | ({group} if group else set())

if hasattr(pre, "feature_names_in_"):
    feature_cols = list(pre.feature_names_in_)
    if any(c in drop for c in feature_cols):
        raise ValueError("Preprocessor expects target/excluded columns")
else:
    cand = [c for c in train.columns if c not in drop]
    keylike = []
    for c in cand:
        s = train[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            if s.nunique(dropna=False) == len(s):
                keylike.append(c)
        elif pd.api.types.is_integer_dtype(s):
            if s.nunique(dropna=False) == len(s):
                keylike.append(c)
    feature_cols = [c for c in cand if c not in keylike]

missing_cols = [c for c in feature_cols if c not in train.columns]
if missing_cols:
    raise KeyError(f"Missing feature columns: {missing_cols}")
X = train[feature_cols]

if isinstance(pre, Pipeline):
    steps = list(pre.steps)
    sel_idx = None
    for i, (name, est) in enumerate(steps):
        if name == "select" or isinstance(est, SelectKBest):
            sel_idx = i
            break
    if sel_idx is not None:
        if sel_idx != len(steps) - 1:
            raise ValueError("Existing selector is not the final step")
        before_steps = steps[:sel_idx]
        before = Pipeline(before_steps) if before_steps else None
        full_steps = steps[:sel_idx] + [("select", None)]
    else:
        before = pre
        full_steps = steps + [("select", None)]
else:
    before = pre
    full_steps = [("preprocess", pre), ("select", None)]

if before is None:
    n_features = X.shape[1]
else:
    try:
        n_features = len(before.get_feature_names_out(feature_cols))
    except Exception:
        n_features = before.transform(X.head(5)).shape[1]

k = min(20, n_features)
sel = SelectKBest(score_func=mutual_info_classif, k=k)
full_steps[-1] = ("select", sel)
full = Pipeline(full_steps)

mlca_runtime.fit_record(full, X, "train", y=y)
selected = int(sel.get_support().sum())

mlca_runtime.record_metric("n_train_rows", int(train.shape[0]), "train")
mlca_runtime.record_metric("n_raw_features", len(feature_cols), "train")
mlca_runtime.record_metric("n_preprocessed_features", n_features, "train")
mlca_runtime.record_metric("n_selected_features", selected, "train")
mlca_runtime.save_model(full, "selector.pkl")

print(f"train rows={train.shape[0]} raw_features={len(feature_cols)} preprocessed_features={n_features} k={k} selected={selected}")
