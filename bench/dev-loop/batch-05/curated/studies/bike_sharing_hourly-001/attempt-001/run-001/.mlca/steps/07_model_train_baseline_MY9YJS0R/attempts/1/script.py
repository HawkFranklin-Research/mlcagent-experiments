import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    group = protocol.get("group_column")
    task = protocol["task"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available: {list(protocol.keys())}")
metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

feat_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    if (s.dtype == object or pd.api.types.is_integer_dtype(s)) and s.nunique(dropna=True) == len(s):
        continue
    feat_cols.append(c)

X_train = train[feat_cols]
y_train = train[targets]
X_val = val[feat_cols]
y_val = val[targets]
print(f"train {X_train.shape} val {X_val.shape} features {len(feat_cols)}")

with open(inputs["05_preprocess_fit_preprocessing_VM7ERF98/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_RGQ2HW1B/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def first_input_names(obj):
    if isinstance(obj, Pipeline) and obj.named_steps:
        obj = next(iter(obj.named_steps.values()))
    return list(obj.feature_names_in_) if hasattr(obj, "feature_names_in_") else None

sel_in = first_input_names(sel)
pre_out_cols = None
try:
    pre_out = pre.transform(X_train.head(5))
    if hasattr(pre_out, "columns"):
        pre_out_cols = set(pre_out.columns)
except Exception:
    pre_out_cols = None

raw_set = set(feat_cols)
if sel_in is not None and pre_out_cols is not None and set(sel_in) <= pre_out_cols:
    steps = [("pre", pre), ("sel", sel)]
elif sel_in is not None and set(sel_in) <= raw_set:
    steps = [("sel", sel)]
else:
    steps = [("pre", pre), ("sel", sel)]

reg = TransformedTargetRegressor(regressor=LinearRegression(),
                                 func=np.log1p, inverse_func=np.expm1)
pipeline = Pipeline(steps + [("lr", reg)])

fitted = mlca_runtime.fit_record(pipeline, X_train, "train", y=y_train)

y_pred = np.asarray(fitted.predict(X_val), dtype=float).reshape(len(X_val), -1)
y_true = np.asarray(y_val, dtype=float).reshape(len(X_val), -1)

rmses = []
for i, t in enumerate(targets):
    r = float(np.sqrt(np.mean((y_true[:, i] - y_pred[:, i]) ** 2)))
    rmses.append(r)
    mlca_runtime.record_metric(f"{metric}_{t}", r, "val")
headline = float(np.mean(rmses))
mlca_runtime.record_metric(metric, headline, "val")
print(f"val {metric} {headline:.4f}")

mlca_runtime.save_model(fitted, "model.pkl")
