import json
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def require(key):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {list(protocol)}")
    return protocol[key]

targets = require("targets")
excluded = require("excluded_columns")
task = require("task")
_ = task.get("primary_metric")
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
for col in targets:
    if col not in train.columns:
        raise KeyError(f"Target column {col!r} not found in training data")

excluded = [c for c in excluded if c in train.columns]

def is_key_like(series):
    n = len(series)
    if n == 0:
        return False
    nunique = series.nunique(dropna=True)
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        return nunique == n
    if pd.api.types.is_integer_dtype(series):
        name = str(series.name).lower()
        return nunique == n and any(tok in name for tok in ("id", "key", "code", "index", "row"))
    return False

feature_cols = []
for col in train.columns:
    if col in targets or col in excluded or (group is not None and col == group):
        continue
    if is_key_like(train[col]):
        continue
    feature_cols.append(col)

if not feature_cols:
    raise ValueError("No feature columns remain after exclusions")

X = train[feature_cols].copy()
y = train[targets].copy()

pre = ColumnTransformer(
    [("keep", "passthrough", feature_cols)],
    remainder="drop",
    verbose_feature_names_out=False,
)
pre.set_output(transform="pandas")

pipe = Pipeline([
    ("pre", pre),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
])
pipe.set_output(transform="pandas")

fitted = mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(fitted, "alternative_model.pkl")

print(f"features={len(feature_cols)} targets={targets}")
print(f"train_rows={len(X)}")
