import mlca_runtime
import json
import pandas as pd
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol["group_column"]
train = pd.read_parquet(inputs["train"])

protected = set(targets) | set(excluded)
if group_col is not None:
    protected.add(group_col)

def is_key_like(s: pd.Series) -> bool:
    if s.isna().any():
        return False
    n = len(s)
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=False) == n
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == n and (s.is_monotonic_increasing or s.is_monotonic_decreasing)
    return False

feature_cols = [
    c for c in train.columns
    if c not in protected and not is_key_like(train[c])
]

selector = ColumnTransformer(
    transformers=[(f"keep_{i}", "passthrough", [c]) for i, c in enumerate(feature_cols)],
    remainder="drop",
    verbose_feature_names_out=False,
)
selector.set_output(transform="pandas")

X = train.loc[:, feature_cols]
selector = mlca_runtime.fit_record(selector, X, "train")
mlca_runtime.save_model(selector, "selector_all.pkl")

print(f"feature_columns={len(feature_cols)}")
print(f"train_rows={len(train)}")
print(f"selector_input_columns={len(selector.feature_names_in_)}")
