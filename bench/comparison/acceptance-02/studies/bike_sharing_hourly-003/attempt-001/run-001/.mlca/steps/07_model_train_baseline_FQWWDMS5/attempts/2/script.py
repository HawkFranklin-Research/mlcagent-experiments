import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in proto:
        raise KeyError(f"Missing {k!r}; available: {list(proto)}")
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group_col = proto.get("group_column")
primary_metric = proto["task"]["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inputs["05_preprocess_fit_preprocessing_SMPGAWB4/preprocessor.pkl"])
sel = load_pkl(inputs["06_features_retain_features_CANXQVVF/selector_all.pkl"])
sel_is_raw = hasattr(sel, "feature_names_in_") and set(sel.feature_names_in_).issubset(set(train.columns))

steps = []
if sel_is_raw:
    steps.append(("selector", sel))
    frontend = sel
else:
    steps.append(("preprocess", pre))
    frontend = pre
    inc = False
    if hasattr(pre, "get_feature_names_out") and hasattr(sel, "feature_names_in_"):
        try:
            inc = set(sel.feature_names_in_).issubset(set(pre.get_feature_names_out()))
        except Exception:
            inc = False
    if inc:
        steps.append(("select", sel))
steps.append(("model", LinearRegression()))

if hasattr(frontend, "feature_names_in_"):
    feature_cols = list(frontend.feature_names_in_)
else:
    drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
    feature_cols = [c for c in train.columns if c not in drop]
    for c in list(feature_cols):
        s = train[c]
        if s.nunique(dropna=False) == len(train) and (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_integer_dtype(s)
        ):
            feature_cols.remove(c)

fitted = mlca_runtime.fit_record(Pipeline(steps), train[feature_cols], "train", y=train[targets])

pred = np.asarray(fitted.predict(val[feature_cols]))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
pred = np.clip(pred, 1e-10, None)

def dev(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-10, None)
    t = np.where(y > 0, y * np.log(y / p) - y, 0.0)
    return float(np.mean(2.0 * (t + p)))

scores = []
for j, t in enumerate(targets):
    s = dev(val[t].to_numpy(), pred[:, j])
    scores.append(s)
    mlca_runtime.record_metric(f"{primary_metric}_{t}", s, "val")
mean_score = float(np.mean(scores))
mlca_runtime.record_metric(primary_metric, mean_score, "val")

mlca_runtime.save_model(fitted, "model.pkl")
print(f"train rows: {len(train)}, features: {len(feature_cols)}")
print(f"val {primary_metric}: {mean_score:.4f}")
