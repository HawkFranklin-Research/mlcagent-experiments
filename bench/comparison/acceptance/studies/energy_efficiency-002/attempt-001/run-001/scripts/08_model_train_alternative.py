import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing {key}; available keys: {list(d)}")
    return d[key]

targets = list(need(proto, "targets"))
excluded = list(need(proto, "excluded_columns"))
group = proto.get("group_column", None)
task = need(proto, "task")
need(task, "primary_metric")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def is_key_like(s):
    n = s.nunique(dropna=True)
    if n <= 1 or n != len(s.dropna()):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        return True
    return False

feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group and not is_key_like(train[c])
]

X_train = train[feature_cols]
y_train = train[targets]

pre = ColumnTransformer(
    [("keep", "passthrough", feature_cols)],
    remainder="drop",
).set_output(transform="pandas")

pipe = Pipeline([
    ("pre", pre),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
])

pipe = mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
mlca_runtime.save_model(pipe, "alternative_model.pkl")

X_val = val[feature_cols]
y_val = val[targets]
pred = pipe.predict(X_val)
scores = []
for i, target in enumerate(targets):
    rmse = root_mean_squared_error(y_val[target], pred[:, i])
    mlca_runtime.record_metric(f"val_rmse_{target}", float(rmse), "val")
    scores.append(float(rmse))
mlca_runtime.record_metric("val_rmse_mean", float(np.mean(scores)), "val")
print(len(feature_cols), X_train.shape, round(float(np.mean(scores)), 4))
