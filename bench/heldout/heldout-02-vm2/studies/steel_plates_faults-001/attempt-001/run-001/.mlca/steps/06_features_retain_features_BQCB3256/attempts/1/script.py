import json
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(key):
    if key not in protocol:
        raise KeyError(f"Missing {key!r}; available top-level keys: {sorted(protocol)}")
    return protocol[key]

targets = list(need("targets"))
excluded = list(need("excluded_columns"))
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
drop = set(targets) | set(excluded)
if group_col is not None:
    drop.add(group_col)

feature_cols = []
for col in train.columns:
    if col in drop:
        continue
    s = train[col]
    n = len(s)
    uniq = s.nunique(dropna=True)
    is_text = (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )
    if is_text and uniq == n:
        continue
    if pd.api.types.is_integer_dtype(s):
        vals = set(s.dropna().unique())
        if uniq == n and not vals.issubset({0, 1}):
            continue
    feature_cols.append(col)

selector = ColumnTransformer(
    [("passthrough", "passthrough", feature_cols)],
    remainder="drop",
    verbose_feature_names_out=False,
)
selector.set_output(transform="pandas")
selector = mlca_runtime.fit_record(selector, train[feature_cols], partition="train")
mlca_runtime.save_model(selector, "selector_all.pkl")
print("features", len(feature_cols))
