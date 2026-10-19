import mlca_runtime
import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = list(proto["targets"])
excluded = list(proto.get("excluded_columns", []))
group = proto.get("group_column")
task = proto["task"]
print("targets:", targets, "excluded:", excluded, "group:", group)

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

feat = [c for c in train.columns if c not in targets and c not in excluded and c != group and c != "dteday"]
keep = []
for c in feat:
    s = train[c]
    if s.dtype.kind in "OU" and s.nunique(dropna=True) == len(s):
        continue
    if s.dtype.kind in "iu" and s.nunique(dropna=True) == len(s) and not set(s.dropna().unique()).issubset({0, 1}):
        continue
    if s.dtype.kind == "M":
        continue
    keep.append(c)
feat = keep

cat_cols, num_cols = [], []
for c in feat:
    s = train[c]
    vals = set(s.dropna().unique())
    is_indicator = vals.issubset({0, 1}) and s.dtype.kind in "iu"
    if not is_indicator and s.dtype.kind in "iu" and s.nunique(dropna=True) <= 24:
        cat_cols.append(c)
    else:
        num_cols.append(c)
print(f"features={len(feat)} numeric={len(num_cols)} categorical={len(cat_cols)}")

pre = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ],
    remainder="drop",
)
try:
    pre.set_output(transform="pandas")
except Exception:
    pass

rf = RandomForestRegressor(n_estimators=100, random_state=42)
if len(targets) == 1:
    rf = TransformedTargetRegressor(regressor=rf, func=np.log1p, inverse_func=np.expm1)

pipe = Pipeline([("prep", pre), ("model", rf)])

X_train = train[feat]
y_train = train[targets[0]] if len(targets) == 1 else train[targets]
fitted = mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)

X_val = val[feat]
if len(targets) == 1:
    y_val = val[targets[0]]
    pred = fitted.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
    mlca_runtime.record_metric("rmse", rmse, "val")
    print(f"val rmse={rmse:.4f}")
else:
    pred = fitted.predict(X_val)
    rmses = []
    for i, t in enumerate(targets):
        r = float(np.sqrt(mean_squared_error(val[t], pred[:, i])))
        rmses.append(r)
        mlca_runtime.record_metric(f"rmse_{t}", r, "val")
    mlca_runtime.record_metric("rmse_mean", float(np.mean(rmses)), "val")
    print(f"val rmse_mean={np.mean(rmses):.4f}")

mlca_runtime.save_model(fitted, "alternative_model.pkl")
print("saved alternative_model.pkl")
