import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
targets = proto["targets"]
excluded = proto.get("excluded_columns", [])
group = proto.get("group_column")
task = proto["task"]
primary = task["primary_metric"]
direction = "min" if primary in ("rmse", "mae", "mse", "log_loss") else "max"

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

feat = [c for c in train.columns if c not in targets and c not in excluded and c != group]
for c in list(feat):
    s = train[c]
    drop = False
    if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
        drop = s.nunique(dropna=False) == len(s)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=False) == len(s):
            u = np.sort(s.dropna().unique())
            drop = len(u) == len(s) and np.array_equal(u, np.arange(u.min(), u.min() + len(u)))
    if c.lower() in ("id", "index") and s.nunique(dropna=False) == len(s):
        drop = True
    if drop and c in feat:
        feat.remove(c)

with open(inp["05_preprocess_fit_preprocessing_RHXAHV44/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_retain_features_W82TE9KX/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

candidates = {}
for key, name in [
    ("07_model_train_baseline_1GT1TPB8/model.pkl", "baseline"),
    ("08_model_train_alternative_KVXQRP8Q/alternative_model.pkl", "alternative"),
    ("latest_model", "latest"),
]:
    p = inp[key]
    if p.resolve() not in {v[0].resolve() for v in candidates.values()}:
        with open(p, "rb") as f:
            candidates[name] = (p, pickle.load(f))

def predict_candidate(obj, X):
    try:
        return obj.predict(X)
    except Exception:
        pass
    if not isinstance(obj, Pipeline):
        for steps in (
            [("pre", pre), ("sel", sel), ("model", obj)],
            [("sel", sel), ("model", obj)],
            [("pre", pre), ("model", obj)],
        ):
            try:
                return Pipeline(steps).predict(X)
            except Exception:
                continue
    raise RuntimeError("candidate cannot predict raw validation features")

X_val = val[feat]
rows = []
best_name = None
best_score = None
for name, (_, obj) in candidates.items():
    pred = np.asarray(predict_candidate(obj, X_val))
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if pred.shape[1] != len(targets):
        raise ValueError(f"prediction shape {pred.shape} does not match targets {targets}")
    y = val[targets].to_numpy()
    metrics = {}
    for i, t in enumerate(targets):
        yt, pt = y[:, i], pred[:, i]
        metrics[f"rmse_{t}"] = float(np.sqrt(mean_squared_error(yt, pt)))
        metrics[f"mae_{t}"] = float(mean_absolute_error(yt, pt))
        metrics[f"r2_{t}"] = float(r2_score(yt, pt))
    for m in ("rmse", "mae", "r2"):
        metrics[f"{m}_macro"] = float(np.mean([metrics[f"{m}_{t}"] for t in targets]))
    for k, v in metrics.items():
        rows.append({"model": name, "metric": k, "value": float(f"{v:.4g}")})
    score = metrics[f"{primary}_macro"]
    if best_score is None or (score < best_score if direction == "min" else score > best_score):
        best_name, best_score = name, score

result = pd.DataFrame(rows)
mlca_runtime.save_table(result, "validation_metrics.csv", caption="Validation metrics per model and target")
mlca_runtime.record_metric(primary, float(best_score), "val")
mlca_runtime.record_fact_candidate("best_validation_score", float(best_score), source="validation_metrics.csv")
print("models evaluated:", len(candidates))
print("primary metric:", primary)
print("best model:", best_name)
print("best validation score:", best_score)
