import mlca_runtime, json
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
for k in ("targets", "excluded_columns", "group_column", "task"):
    if k not in protocol:
        raise KeyError(f"missing {k}; available: {sorted(protocol)}")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

drop = set(targets) | set(excluded) | ({group} if group is not None else set())
features = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if s.is_unique and (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_integer_dtype(s)
    ):
        continue
    features.append(c)
if not features:
    raise ValueError("no feature columns after exclusions")

X = train[features].copy()
y = train[targets].copy()

num = [c for c in features if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
max_cat = max(5, int(0.05 * len(X)))
cat = [c for c in features if c not in num and X[c].nunique(dropna=False) <= max_cat]

parts = []
if num:
    parts.append(("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]), num))
if cat:
    parts.append(("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), cat))
if not parts:
    raise ValueError("no usable feature columns for preprocessing")

ct = ColumnTransformer(parts, remainder="drop")
try:
    ct.set_output(transform="pandas")
except Exception:
    pass

model = Pipeline([("pre", ct), ("model", LinearRegression())])
fitted = mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(fitted, "model.pkl")

Xv = val[features].copy()
yv = val[targets].to_numpy()
pred = np.asarray(fitted.predict(Xv))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)

for j, t in enumerate(targets):
    rmse = float(np.sqrt(np.mean((pred[:, j] - yv[:, j]) ** 2)))
    mlca_runtime.record_metric(f"val_rmse_{t}", rmse, "val")
headline = float(np.sqrt(np.mean((pred - yv) ** 2)))
mlca_runtime.record_metric("val_rmse", headline, "val")

print(f"features={len(features)} num={len(num)} cat={len(cat)}")
print(f"train_rows={len(train)} val_rows={len(val)}")
print(f"val_rmse={headline:.6f}")
