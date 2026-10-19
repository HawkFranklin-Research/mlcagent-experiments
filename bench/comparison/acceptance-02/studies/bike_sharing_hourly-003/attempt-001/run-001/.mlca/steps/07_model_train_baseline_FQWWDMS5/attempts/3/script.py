import mlca_runtime
import json, pickle
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

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load(inputs["05_preprocess_fit_preprocessing_SMPGAWB4/preprocessor.pkl"])
sel = load(inputs["06_features_retain_features_CANXQVVF/selector_all.pkl"])

# Ensure pandas output when downstream steps expect feature names
for obj in (pre, sel):
    if hasattr(obj, "set_output"):
        try:
            obj.set_output(transform="pandas")
        except Exception:
            pass
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            if hasattr(step, "set_output"):
                try:
                    step.set_output(transform="pandas")
                except Exception:
                    pass

def has_raw(obj):
    return hasattr(obj, "feature_names_in_") and set(obj.feature_names_in_).issubset(set(train.columns))

steps = []
if has_raw(sel) and isinstance(sel, Pipeline) and len(sel.named_steps) > 1:
    steps = [("sel", sel)]
    feature_cols = list(sel.feature_names_in_)
elif has_raw(pre):
    steps = [("pre", pre)]
    feature_cols = list(pre.feature_names_in_)
    if hasattr(sel, "feature_names_in_"):
        try:
            out = pre.get_feature_names_out()
            if set(sel.feature_names_in_).issubset(set(out)):
                steps.append(("sel", sel))
        except Exception:
            pass
else:
    drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
    feature_cols = [c for c in train.columns if c not in drop]
    for c in list(feature_cols):
        s = train[c]
        if s.nunique(dropna=False) == len(train) and (
            pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_integer_dtype(s)
        ):
            feature_cols.remove(c)
    steps = [("pre", pre)] if pre is not None else []

drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
feature_cols = [c for c in feature_cols if c not in drop]
steps.append(("model", LinearRegression()))
pipe = Pipeline(steps)

fitted = mlca_runtime.fit_record(pipe, train[feature_cols], "train", y=train[targets])

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
