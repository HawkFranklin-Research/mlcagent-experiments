import json
import mlca_runtime
import pandas as pd
from pandas.api.types import is_integer_dtype, is_string_dtype, is_object_dtype
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for k in ("targets", "excluded_columns"):
    if k not in protocol:
        raise KeyError(f"protocol missing {k!r}; available: {sorted(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

n = len(train)
feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    nun = s.nunique(dropna=True)
    if n > 1 and nun == n and (is_integer_dtype(s) or is_string_dtype(s) or is_object_dtype(s)):
        continue
    feature_cols.append(c)

X = train[feature_cols]

selector = ColumnTransformer(
    [("all", "passthrough", feature_cols)],
    remainder="drop",
    verbose_feature_names_out=False,
)
selector.set_output(transform="pandas")
selector = mlca_runtime.fit_record(selector, X, "train")

mlca_runtime.save_model(selector, "selector_all.pkl")
print(f"n_features={len(feature_cols)} n_rows={len(X)}")
