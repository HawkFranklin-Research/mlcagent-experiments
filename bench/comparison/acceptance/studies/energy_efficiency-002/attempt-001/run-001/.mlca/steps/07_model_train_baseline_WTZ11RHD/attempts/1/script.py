import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = list(proto["targets"])
excluded = list(proto.get("excluded_columns") or [])
group = proto.get("group_column")
task = proto["task"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded) | ({group} if group else set())
feat_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if s.dtype == object:
        n = int(s.notna().sum())
        if n and int(s.nunique()) == n:
            continue
    elif np.issubdtype(s.dtype, np.integer):
        n = int(s.notna().sum())
        if n and int(s.nunique()) == n and s.nunique() > 10:
            continue
    feat_cols.append(c)

with open(inputs["05_preprocess_fit_preprocessing_9HEDMWQR/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_HD824TRJ/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def has_preprocessing_inside(obj):
    if isinstance(obj, ColumnTransformer):
        return True
    if isinstance(obj, Pipeline):
        for _, st in obj.steps:
            if has_preprocessing_inside(st):
                return True
    return False

def required_inputs(obj):
    if isinstance(obj, Pipeline):
        for _, st in obj.steps:
            r = required_inputs(st)
            if r is not None:
                return r
        return None
    return list(obj.feature_names_in_) if hasattr(obj, "feature_names_in_") else None

sel_is_empty = sel is None or (isinstance(sel, str) and sel.lower() in ("passthrough", "none"))
sel_is_full = isinstance(sel, Pipeline) and has_preprocessing_inside(sel)

if sel_is_empty:
    front_steps = [("frontend", pre)]
elif sel_is_full:
    front_steps = [("frontend", sel)]
else:
    front_steps = [("pre", pre), ("sel", sel)]

pipeline = Pipeline(front_steps + [("lr", LinearRegression())])
frontend = front_steps[0][1]
req = required_inputs(frontend)
X = train[req] if req is not None else train[feat_cols]
y = train[targets]

fitted = mlca_runtime.fit_record(pipeline, X, partition="train", y=y)
mlca_runtime.save_model(fitted, "model.pkl")

Xv = val[list(X.columns)]
yv = val[targets].to_numpy()
pred = fitted.predict(Xv)
rmses = np.sqrt(((pred - yv) ** 2).mean(axis=0))
for i, t in enumerate(targets):
    mlca_runtime.record_metric(f"rmse_{t}", float(rmses[i]), split="val")
mlca_runtime.record_metric("rmse_mean", float(rmses.mean()), split="val")
print(f"train rows={len(train)} val rows={len(val)}")
print(f"features={len(X.columns)} targets={targets}")
print(f"val rmse per target: {dict(zip(targets, np.round(rmses,4)))}")
print(f"val rmse mean: {rmses.mean():.4f}")
