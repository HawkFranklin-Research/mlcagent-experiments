import json
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)

paths = mlca_runtime.inputs()
train = pd.read_parquet(paths["train"])
protocol = json.loads(paths["_protocol"].read_text())

def required(d, key):
    if key not in d:
        raise KeyError(f"Missing required key {key!r}; available keys: {sorted(d)}")
    return d[key]

targets = list(required(protocol, "targets"))
excluded = set(required(protocol, "excluded_columns"))
group_col = protocol.get("group_column")
if group_col:
    excluded.add(group_col)

def key_like(s):
    if pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s) and s.is_monotonic_increasing
    return False

feature_cols = [
    c for c in train.columns
    if c not in set(targets) and c not in excluded and not key_like(train[c])
]
if not feature_cols:
    raise ValueError("No feature columns remain after applying protocol exclusions.")

X = train.loc[:, feature_cols]

selector = ColumnTransformer(
    [("passthrough", "passthrough", feature_cols)],
    remainder="drop",
    verbose_feature_names_out=False,
).set_output(transform="pandas")

selector = mlca_runtime.fit_record(selector, X, "train")

out_names = list(selector.get_feature_names_out())
if out_names != feature_cols:
    raise RuntimeError("Feature names or order were not preserved by the selector.")

mlca_runtime.save_model(selector, "selector_all.pkl")
print("saved selector_all.pkl")
