import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol.get("targets")
if targets is None:
    raise KeyError(f"Missing 'targets' in _protocol. Available keys: {list(protocol.keys())}")
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"]) if "val" in inputs else None

with open(inputs["05_preprocess_fit_preprocessing_JT5JG4F1/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_retain_features_6CDSH3YD/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)

def wraps_prep(obj):
    if not isinstance(obj, Pipeline):
        return False
    return any(isinstance(step, (ColumnTransformer, Pipeline)) for step in obj.named_steps.values())

if isinstance(selector, Pipeline) and wraps_prep(selector):
    frontend = selector
    feature_cols = list(frontend.feature_names_in_) if hasattr(frontend, 'feature_names_in_') else None
    steps = [('frontend', frontend)]
else:
    steps = []
    if preprocessor is not None:
        steps.append(('preprocessor', preprocessor))
    if selector is not None:
        steps.append(('selector', selector))
    feature_cols = None
    if preprocessor is not None and hasattr(preprocessor, 'feature_names_in_'):
        feature_cols = list(preprocessor.feature_names_in_)

if feature_cols is None:
    def is_key_like(s):
        if s.dtype == 'object' or s.dtype.name == 'category':
            return s.nunique() == len(s)
        if pd.api.types.is_integer_dtype(s):
            return s.nunique() == len(s)
        return False
    all_cols = train.columns.tolist()
    feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col and not is_key_like(train[c])]

feature_cols = [c for c in feature_cols if c not in targets and c not in excluded and c != group_col]
steps.append(('model', LinearRegression()))
pipeline = Pipeline(steps)

X_train = train[feature_cols].copy()
y_train = train[targets].copy()

fitted = mlca_runtime.fit_record(pipeline, X_train, "train", y_train)
mlca_runtime.save_model(fitted, "model.pkl")

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

train_pred = fitted.predict(X_train)
if isinstance(y_train, pd.DataFrame):
    rmses = [rmse(y_train[col], train_pred[:, i]) for i, col in enumerate(y_train.columns)]
    for i, col in enumerate(y_train.columns):
        mlca_runtime.record_metric(f"rmse_{col}_train", rmses[i], "train")
    headline = float(np.mean(rmses))
    mlca_runtime.record_metric("rmse_mean_train", headline, "train")
else:
    headline = rmse(y_train, train_pred)
    mlca_runtime.record_metric("rmse_train", headline, "train")

if val is not None:
    X_val = val[feature_cols].copy()
    y_val = val[targets].copy()
    val_pred = fitted.predict(X_val)
    if isinstance(y_val, pd.DataFrame):
        rmses_val = [rmse(y_val[col], val_pred[:, i]) for i, col in enumerate(y_val.columns)]
        for i, col in enumerate(y_val.columns):
            mlca_runtime.record_metric(f"rmse_{col}_val", rmses_val[i], "val")
        headline_val = float(np.mean(rmses_val))
        mlca_runtime.record_metric("rmse_mean_val", headline_val, "val")
    else:
        headline_val = rmse(y_val, val_pred)
        mlca_runtime.record_metric("rmse_val", headline_val, "val")

print(f"Train shape: {train.shape}, Val shape: {val.shape if val is not None else 'N/A'}")
print(f"Features: {len(feature_cols)}")
print(f"Train RMSE: {headline:.4f}")
if val is not None:
    print(f"Val RMSE: {headline_val:.4f}")
