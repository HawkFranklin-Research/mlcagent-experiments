import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing _protocol[{k!r}]; available top-level keys: {sorted(d)}")
    return d[k]

targets = list(need(protocol, "targets"))
excluded = list(need(protocol, "excluded_columns"))
group = need(protocol, "group_column")
_ = need(protocol, "task")

train = pd.read_parquet(inputs["train"])
drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

def keylike(s):
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        return bool(s.notna().all() and s.nunique(dropna=True) == len(s))
    return False

feature_cols = [c for c in train.columns if c not in drop and not keylike(train[c])]

with open(inputs["05_preprocess_fit_preprocessing_KTEB41J1/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

raw_cols = list(pre.feature_names_in_) if hasattr(pre, "feature_names_in_") else feature_cols
bad = [c for c in raw_cols if c in drop]
if bad:
    raise ValueError(f"Preprocessor input includes excluded/target columns: {bad}")
missing = [c for c in raw_cols if c not in train.columns]
if missing:
    raise KeyError(f"Missing preprocessor input columns: {missing}")

X_raw = train[raw_cols]
X_proc = pre.transform(X_raw)

if hasattr(X_proc, "toarray"):
    X_proc = X_proc.toarray()

if isinstance(X_proc, pd.DataFrame):
    proc_cols = list(X_proc.columns)
    X_proc_df = X_proc.copy()
else:
    if not hasattr(pre, "get_feature_names_out"):
        raise ValueError("Preprocessor output has no feature names")
    proc_cols = list(pre.get_feature_names_out())
    X_proc_df = pd.DataFrame(X_proc, columns=proc_cols, index=train.index)

X_proc_df.index = train.index
if X_proc_df.shape[1] != len(proc_cols):
    raise ValueError("Preprocessor output width does not match feature names")

selector = ColumnTransformer(
    transformers=[("passthrough", "passthrough", proc_cols)],
    remainder="drop",
    verbose_feature_names_out=False,
)
selector.set_output(transform="pandas")

fitted = mlca_runtime.fit_record(selector, X_proc_df, "train")
out_names = list(fitted.get_feature_names_out())
if out_names != proc_cols:
    raise ValueError("Selector did not preserve feature names and order")

mlca_runtime.save_model(fitted, "selector_all.pkl")
mlca_runtime.record_metric("n_features_selected", float(len(proc_cols)), "train")
mlca_runtime.record_metric("n_train_rows", float(len(train)), "train")

print(f"train rows: {len(train)}")
print(f"preprocessor input columns: {len(raw_cols)}")
print(f"selected features: {len(proc_cols)}")
print("saved selector_all.pkl")
