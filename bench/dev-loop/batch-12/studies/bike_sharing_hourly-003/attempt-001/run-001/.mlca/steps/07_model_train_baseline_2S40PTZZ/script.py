import mlca_runtime, json
import pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
task = protocol["task"]
metric = task["primary_metric"]
time_cols = set(task.get("time_columns", []) or [])

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded) | time_cols
if group_col:
    drop.add(group_col)

feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if s.dtype == object:
        if s.nunique() == len(s):
            continue
        try:
            pd.to_datetime(s.dropna().head(5))
            continue
        except Exception:
            pass
    feature_cols.append(c)

numeric_cols, cat_cols = [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    else:
        cat_cols.append(c)

n_enc = len(numeric_cols) + sum(int(train[c].nunique()) + 1 for c in cat_cols)

pre_transformers = [("num", SimpleImputer(strategy="median"), numeric_cols)]
if cat_cols:
    pre_transformers.append((
        "cat",
        Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=(n_enc > 300))),
        ]),
        cat_cols,
    ))

preproc = ColumnTransformer(pre_transformers, remainder="drop")
if n_enc <= 300:
    preproc.set_output(transform="pandas")

pipeline = Pipeline([("pre", preproc), ("model", LinearRegression())])

X_tr = train[feature_cols]
y_tr = train[targets].astype(float)
fitted = mlca_runtime.fit_record(pipeline, X_tr, partition="train", y=y_tr)

X_val = val[feature_cols]
pred = fitted.predict(X_val)
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
y_val = val[targets].astype(float).values
rmse_per = [float(np.sqrt(np.mean((pred[:, i] - y_val[:, i]) ** 2))) for i in range(len(targets))]
headline = float(np.mean(rmse_per))

for t, v in zip(targets, rmse_per):
    mlca_runtime.record_metric(f"{metric}_{t}", v, "val")
mlca_runtime.record_metric(metric, headline, "val")

mlca_runtime.save_model(fitted, "model.pkl")
print(f"features={len(feature_cols)} numeric={len(numeric_cols)} cat={len(cat_cols)} n_enc={n_enc}")
print(f"val_{metric}_mean={headline:.4f} per_target={[round(v, 4) for v in rmse_per]}")
