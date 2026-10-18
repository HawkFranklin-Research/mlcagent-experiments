import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_XD81KYMB/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
_ = json.loads(inputs["05_preprocess_fit_preprocessing_XD81KYMB/manifest.json"].read_text())

required = ["targets", "excluded_columns", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
group_cols = [] if group is None else ([group] if isinstance(group, str) else list(group))
drop = set(targets) | set(excluded) | set(group_cols)

for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
        drop.add(c)

feature_cols = [c for c in train.columns if c not in drop]
if hasattr(pre, "feature_names_in_") and pre.feature_names_in_ is not None:
    raw_cols = list(pre.feature_names_in_)
else:
    raw_cols = feature_cols

bad = set(raw_cols) & (set(targets) | set(excluded) | set(group_cols))
if bad:
    raise ValueError(f"Preprocessor expects excluded columns: {bad}")
missing_cols = [c for c in raw_cols if c not in train.columns]
if missing_cols:
    raise KeyError(f"Missing columns in train: {missing_cols}")

X = train[raw_cols].copy()
if hasattr(pre, "set_output"):
    try:
        pre.set_output(transform="pandas")
    except Exception:
        pass

X_trans = pre.transform(X)
selector = FunctionTransformer()
if hasattr(selector, "set_output"):
    try:
        selector.set_output(transform="pandas")
    except Exception:
        pass
selector = mlca_runtime.fit_record(selector, X_trans, partition="train", y=None)

if isinstance(pre, Pipeline):
    steps = list(pre.steps) + [("passthrough_selector", selector)]
    pipe = Pipeline(steps)
else:
    pipe = Pipeline([("preprocessor", pre), ("passthrough_selector", selector)])
if hasattr(pipe, "set_output"):
    try:
        pipe.set_output(transform="pandas")
    except Exception:
        pass

mlca_runtime.save_model(pipe, "selector_all.pkl")
mlca_runtime.record_metric("selector_n_train_rows", len(X), "train")
mlca_runtime.record_metric("selector_n_features", len(raw_cols), "train")
print(f"rows={len(X)} features={len(raw_cols)}")
print("saved selector_all.pkl")
