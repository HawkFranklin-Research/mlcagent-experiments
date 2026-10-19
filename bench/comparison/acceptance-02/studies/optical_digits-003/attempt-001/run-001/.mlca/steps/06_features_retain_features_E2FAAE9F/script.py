import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing {k!r} in _protocol; available top-level keys: {sorted(d.keys())}")
    return d[k]

targets = list(need(protocol, "targets"))
excluded = list(need(protocol, "excluded_columns"))
need(protocol, "task")
group = protocol.get("group_column", None)

train = pd.read_parquet(inputs["train"])
used = set(targets) | set(excluded)
if group is not None:
    if group not in train.columns:
        raise KeyError(f"Declared group column {group!r} is absent from train")
    used.add(group)

feature_cols = [c for c in train.columns if c not in used]

pre = pickle.loads(inputs["05_preprocess_fit_preprocessing_4ZZYYY8Y/preprocessor.pkl"].read_bytes())
manifest = json.loads(inputs["05_preprocess_fit_preprocessing_4ZZYYY8Y/manifest.json"].read_text())

pre_cols = getattr(pre, "feature_names_in_", None)
if pre_cols is not None:
    pre_cols = list(pre_cols)
    missing = set(pre_cols) - set(train.columns)
    if missing:
        raise KeyError(f"Preprocessor expects columns absent from train: {sorted(missing)}")
    forbidden = set(pre_cols) & used
    if forbidden:
        raise ValueError(f"Preprocessor expects excluded/target/group columns: {sorted(forbidden)}")
    X_raw = train.loc[:, pre_cols]
else:
    X_raw = train.loc[:, feature_cols]

X_pre = pre.transform(X_raw)
if hasattr(X_pre, "toarray"):
    X_pre = X_pre.toarray()
if not isinstance(X_pre, pd.DataFrame):
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(X_pre.shape[1])]
    X_pre = pd.DataFrame(X_pre, index=X_raw.index, columns=names)
else:
    X_pre.index = X_raw.index

selector_features = list(X_pre.columns)
selector = ColumnTransformer(
    [("passthrough", "passthrough", selector_features)],
    remainder="drop",
    verbose_feature_names_out=False,
)
mlca_runtime.fit_record(selector, X_pre, partition="train")
mlca_runtime.save_model(selector, "selector_all.pkl")
mlca_runtime.record_metric("selector_n_features", float(len(selector_features)), "train")
mlca_runtime.record_metric("selector_n_rows", float(len(X_pre)), "train")
print(f"train rows={len(train)} raw features={len(feature_cols)} selector features={len(selector_features)}")
print("saved selector_all.pkl")
