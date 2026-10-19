import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing {key!r}; available top-level keys: {sorted(d.keys())}")
    return d[key]

protocol = json.loads(inputs["_protocol"].read_text())
targets = list(need(protocol, "targets"))
excluded = list(need(protocol, "excluded_columns"))
group = need(protocol, "group_column")
task = need(protocol, "task")
if not isinstance(task, dict):
    raise TypeError("_protocol['task'] must be a mapping")

train = pd.read_parquet(inputs["train"])
drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

feature_cols = [c for c in train.columns if c not in drop]
key_cols = []
n = len(train)
for c in feature_cols:
    s = train[c]
    if s.nunique(dropna=False) == n and (
        pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
    ):
        key_cols.append(c)
feature_cols = [c for c in feature_cols if c not in key_cols]

with open(inputs["05_preprocess_fit_preprocessing_MZ4P76F5/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
json.loads(inputs["05_preprocess_fit_preprocessing_MZ4P76F5/manifest.json"].read_text())

pre_cols = getattr(pre, "feature_names_in_", None)
if pre_cols is not None:
    pre_cols = list(pre_cols)
    if set(pre_cols) & drop:
        raise ValueError("Preprocessor expects a target/excluded/group column")
    if set(pre_cols) != set(feature_cols):
        raise ValueError("Preprocessor feature_names_in_ mismatch")
    feature_cols = pre_cols

X = train[feature_cols]
try:
    out_names = list(pre.get_feature_names_out(feature_cols))
except Exception:
    out_names = None

if out_names is not None and len(out_names) <= 300 and hasattr(pre, "set_output"):
    pre.set_output(transform="pandas")

X_trans = pre.transform(X)
if out_names is not None:
    if hasattr(X_trans, "columns"):
        X_trans = X_trans.loc[:, out_names]
    elif not hasattr(X_trans, "toarray") or len(out_names) <= 300:
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        X_trans = pd.DataFrame(X_trans, columns=out_names, index=X.index)

selector = FunctionTransformer(feature_names_out="one-to-one", accept_sparse=True)
y = train[targets[0]] if len(targets) == 1 else train[targets]
selector = mlca_runtime.fit_record(selector, X_trans, partition="train", y=y)
mlca_runtime.save_model(selector, "selector_all.pkl")

print(f"train_rows={len(train)} raw_features={len(feature_cols)} transformed={getattr(X_trans, 'shape', 'unknown')}")
print("saved selector_all.pkl")
