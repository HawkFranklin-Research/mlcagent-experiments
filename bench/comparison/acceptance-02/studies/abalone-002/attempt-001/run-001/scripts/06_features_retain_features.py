import json
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {list(d.keys())}")
    return d[k]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
task = need(protocol, "task")
group = protocol.get("group_column", None)

train = pd.read_parquet(inputs["train"])
drop = set(targets) | set(excluded)
if group:
    drop.add(group)

def is_key_like(s, name):
    if s.nunique(dropna=False) != len(s):
        return False
    lname = name.lower()
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s) and any(t in lname for t in ("id", "key", "index", "code")):
        return True
    return False

feature_cols = [
    c for c in train.columns
    if c not in drop and not is_key_like(train[c], c)
]
if not feature_cols:
    raise ValueError("No feature columns remain")

selector = ColumnTransformer(
    transformers=[("passthrough", "passthrough", feature_cols)],
    remainder="drop",
    verbose_feature_names_out=False,
)
selector.set_output(transform="pandas")

X_train = train.loc[:, feature_cols]
fitted = mlca_runtime.fit_record(selector, X_train, partition="train")
mlca_runtime.save_model(fitted, "selector_all.pkl")

print(f"train rows: {len(train)}")
print(f"feature columns: {len(feature_cols)}")
print("saved selector_all.pkl")
