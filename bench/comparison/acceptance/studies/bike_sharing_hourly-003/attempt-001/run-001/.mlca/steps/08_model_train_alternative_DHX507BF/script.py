import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
task = protocol["task"]

df_train = pd.read_parquet(inputs["train"])
all_cols = list(df_train.columns)

drop = set(targets) | set(excluded)
if group_col is not None and group_col in all_cols:
    drop.add(group_col)

# key-like: text codes or integer sequences with all distinct values
for col in all_cols:
    if col in drop:
        continue
    s = df_train[col]
    if s.dtype == "object" or pd.api.types.is_string_dtype(s.dtype) or pd.api.types.is_integer_dtype(s.dtype):
        if s.nunique() == len(s) and s.nunique() > 1:
            drop.add(col)

feature_cols = [c for c in all_cols if c not in drop]
X_train = df_train[feature_cols].copy()
y_train = df_train[targets].copy()

cat_cols = []
num_cols = []
for c in feature_cols:
    s = df_train[c]
    if s.dtype == "object" or pd.api.types.is_string_dtype(s.dtype) or s.dtype.name == "category":
        cat_cols.append(c)
    elif pd.api.types.is_integer_dtype(s.dtype) and s.nunique() <= 20:
        cat_cols.append(c)
    else:
        num_cols.append(c)

total_cat_width = sum(df_train[c].nunique() for c in cat_cols)
if total_cat_width <= 300:
    sparse_output = False
    set_output_pandas = True
else:
    sparse_output = True
    set_output_pandas = False

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output))
])
preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])
if set_output_pandas:
    preprocessor.set_output(transform="pandas")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])
if set_output_pandas:
    pipeline.set_output(transform="pandas")

model = TransformedTargetRegressor(
    regressor=pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)

fitted = mlca_runtime.fit_record(model, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "alternative_model.pkl")

val_path = inputs.get("val")
if val_path is not None:
    df_val = pd.read_parquet(val_path)
    X_val = df_val[feature_cols].copy()
    y_val = df_val[targets].copy()
    preds = fitted.predict(X_val)
    rmse_per_target = [
        float(np.sqrt(mean_squared_error(y_val.iloc[:, i], preds if preds.ndim == 1 else preds[:, i])))
        for i in range(len(targets))
    ]
    rmse = float(np.mean(rmse_per_target))
    mlca_runtime.record_metric("rmse", rmse, "val")
    print(f"n_features={len(feature_cols)} cat_width={total_cat_width} val_rmse={rmse:.4f}")
else:
    print(f"n_features={len(feature_cols)} cat_width={total_cat_width}")
