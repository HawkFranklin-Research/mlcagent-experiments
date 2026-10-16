import mlca_runtime
import json
import pandas as pd
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def get_key(d, key):
    if key not in d:
        raise KeyError(f"Missing key '{key}'. Available top-level keys: {list(d.keys())}")
    return d[key]

targets = get_key(protocol, "targets")
excluded = get_key(protocol, "excluded_columns")
group = get_key(protocol, "group_column")
task = get_key(protocol, "task")

train = pd.read_parquet(inputs["train"])

def is_key_like(s):
    if s.dtype.kind in "O" or s.dtype.kind in "U" or s.dtype.kind in "S":
        return s.nunique(dropna=False) == len(s) and s.notna().all()
    if s.dtype.kind in "i" or s.dtype.kind in "u":
        return s.nunique(dropna=False) == len(s) and s.notna().all()
    return False

feature_cols = []
for col in train.columns:
    if col in targets or col in excluded or col == group:
        continue
    if is_key_like(train[col]):
        continue
    feature_cols.append(col)

X_train = train[feature_cols].copy()

selector = ColumnTransformer(
    transformers=[("passthrough", "passthrough", feature_cols)],
    remainder="drop",
    verbose_feature_names_out=False,
)
selector.set_output(transform="pandas")

mlca_runtime.fit_record(selector, X_train, "train")
mlca_runtime.save_model(selector, "selector_all.pkl")

mlca_runtime.record_metric("n_features", len(feature_cols), "train")
mlca_runtime.record_metric("n_train_rows", len(X_train), "train")

print(f"Features: {len(feature_cols)}")
print(f"Train rows: {len(X_train)}")
print("Saved selector_all.pkl")
