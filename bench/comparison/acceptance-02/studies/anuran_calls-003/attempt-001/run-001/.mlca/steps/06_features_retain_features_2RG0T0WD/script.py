import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]

train = pd.read_parquet(inputs["train"])

with open(inputs["05_preprocess_fit_preprocessing_7F4HZTDJ/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    raw_cols = list(pre.feature_names_in_)
else:
    drop = set(targets) | set(excluded) | {group}
    raw_cols = []
    n = len(train)
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        if s.dtype.kind in "O" and s.nunique(dropna=True) == n:
            continue
        if s.dtype.kind in "iu" and s.nunique(dropna=True) == n:
            continue
        raw_cols.append(c)

missing_cols = [c for c in raw_cols if c not in train.columns]
if missing_cols:
    raise KeyError(f"Preprocessor expects missing columns: {missing_cols}")

X_raw = train[raw_cols].copy()
X_trans = pre.transform(X_raw)

if hasattr(X_trans, "columns"):
    X = X_trans
else:
    names = pre.get_feature_names_out()
    X = pd.DataFrame(X_trans, columns=names, index=X_raw.index)

selector = ColumnTransformer(transformers=[], remainder="passthrough")
selector.set_output(transform="pandas")
mlca_runtime.fit_record(selector, X, partition="train")
mlca_runtime.save_model(selector, "selector_all.pkl")

mlca_runtime.record_metric("n_features_selected", int(X.shape[1]), "train")
mlca_runtime.record_metric("n_train_rows", int(X.shape[0]), "train")
print(f"rows={X.shape[0]} features={X.shape[1]} saved=selector_all.pkl")
