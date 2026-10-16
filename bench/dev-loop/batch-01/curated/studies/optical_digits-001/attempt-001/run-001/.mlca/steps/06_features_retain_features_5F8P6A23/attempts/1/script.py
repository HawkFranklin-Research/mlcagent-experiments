import json
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def get_key(d, k):
    if k not in d:
        raise KeyError(f"Missing {k!r}. Available top-level keys: {list(d.keys())}")
    return d[k]

targets = get_key(protocol, "targets")
excluded = get_key(protocol, "excluded_columns")
group = get_key(protocol, "group_column")
task = get_key(protocol, "task")

train = pd.read_parquet(inputs["train"])

def is_key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False

group_cols = [] if group is None else ([group] if isinstance(group, str) else list(group))
feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c not in group_cols and not is_key_like(train[c])
]

X_train = train[feature_cols].copy()

selector = ColumnTransformer(
    transformers=[("passthrough", "passthrough", feature_cols)],
    remainder="drop",
    verbose_feature_names_out=False,
).set_output(transform="pandas")

selector = mlca_runtime.fit_record(selector, X_train, "train")
mlca_runtime.save_model(selector, "selector_all.pkl")
mlca_runtime.record_metric("n_features", len(feature_cols), "train")
mlca_runtime.record_metric("n_train_rows", len(X_train), "train")
print(f"train rows: {len(X_train)}")
print(f"features: {len(feature_cols)}")
print("saved selector_all.pkl")
