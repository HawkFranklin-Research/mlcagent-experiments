import json
import pandas as pd
from sklearn.compose import ColumnTransformer
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

def required(d, key):
    if key not in d:
        raise KeyError(f"Missing {key!r} in _protocol; available top-level keys: {sorted(d.keys())}")
    return d[key]

targets = required(protocol, "targets")
excluded = required(protocol, "excluded_columns")
group_col = required(protocol, "group_column")

drop = set(targets) | set(excluded)
if group_col is not None:
    drop.add(group_col)

candidate_cols = [c for c in train.columns if c not in drop]
n = len(train)
feature_cols = []
for c in candidate_cols:
    s = train[c]
    if s.nunique(dropna=False) == n:
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            continue
        if pd.api.types.is_integer_dtype(s) and (s.is_monotonic_increasing or s.is_monotonic_decreasing):
            continue
    feature_cols.append(c)

X_train = train.loc[:, feature_cols]
selector = ColumnTransformer(
    [("keep", "passthrough", feature_cols)],
    remainder="drop",
    verbose_feature_names_out=False,
).set_output(transform="pandas")

fitted = mlca_runtime.fit_record(selector, X_train, "train")
mlca_runtime.save_model(fitted, "selector_all.pkl")
mlca_runtime.record_metric("n_features_selected", float(len(feature_cols)), "train")
print(f"train rows: {len(train)}")
print(f"feature columns: {len(feature_cols)}")
print("saved selector_all.pkl")
