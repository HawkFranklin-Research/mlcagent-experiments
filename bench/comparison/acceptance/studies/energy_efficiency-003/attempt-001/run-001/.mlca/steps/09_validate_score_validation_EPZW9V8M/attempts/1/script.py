import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
protocol = json.loads(inp["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise RuntimeError(f"_protocol missing {k!r}; available={sorted(d)}")
    return d[k]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group_col = need(protocol, "group_column")
task = need(protocol, "task")
primary = need(task, "primary_metric")

protected = set(targets) | set(excluded) | ({group_col} if group_col else set())
feature_cols = []
for c in train.columns:
    if c in protected:
        continue
    s = train[c]
    if s.dtype.kind in "O" and s.nunique(dropna=False) == len(s):
        continue
    if s.dtype.kind in "iu" and s.nunique(dropna=False) == len(s) and any(k in str(c).lower() for k in ("id", "code", "key", "index")):
        continue
    feature_cols.append(c)

def load_pkl(path):
    return pickle.loads(path.read_bytes())

pre = load_pkl(inp["05_preprocess_fit_preprocessing_X2VH4MDF/preprocessor.pkl"])
sel = load_pkl(inp["06_features_retain_features_B66XBBY7/selector_all.pkl"])

def build_candidate(obj):
    if isinstance(obj, Pipeline):
        return obj
    fni = getattr(obj, "feature_names_in_", None)
    if fni is not None and all(c in feature_cols for c in list(fni)):
        return obj
    steps = []
    if pre is not None:
        steps.append(("preprocessor", pre))
    if sel is not None:
        steps.append(("selector", sel))
    steps.append(("model", obj))
    return Pipeline(steps)

paths = [
    ("baseline", inp["07_model_train_baseline_QGQE700X/model.pkl"]),
    ("alternative", inp["08_model_train_alternative_8P21XYS1/alternative_model.pkl"]),
    ("latest", inp["latest_model"]),
]
seen, candidates = set(), []
for name, path in paths:
    rp = path.resolve()
    if rp in seen:
        continue
    seen.add(rp)
    candidates.append((name, build_candidate(load_pkl(path))))

y_val = val[targets].to_numpy()
rows, scores = [], {}

def pred_for(predictor):
    X = val[feature_cols].copy()
    fni = getattr(predictor, "feature_names_in_", None)
    if fni is not None:
        fni = list(fni)
        missing = [c for c in fni if c not in X.columns]
        if missing:
            raise RuntimeError(f"predictor expects missing columns: {missing}")
        X = X[fni]
    p = np.asarray(predictor.predict(X))
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    if p.shape[1] != len(targets):
        raise RuntimeError(f"prediction has {p.shape[1]} columns, expected {len(targets)}")
    return p

for name, predictor in candidates:
    pred = pred_for(predictor)
    per = {}
    for j, t in enumerate(targets):
        yt, yp = y_val[:, j], pred[:, j]
        per[f"rmse_{t}"] = float(np.sqrt(mean_squared_error(yt, yp)))
        per[f"mae_{t}"] = float(mean_absolute_error(yt, yp))
        per[f"r2_{t}"] = float(r2_score(yt, yp))
    for m in ("rmse", "mae", "r2"):
        per[f"{m}_macro"] = float(np.mean([per[f"{m}_{t}"] for t in targets]))
    scores[name] = per
    for k, v in per.items():
        rows.append({"model": name, "metric": k, "value": float(f"{v:.4g}")})

key = {"rmse": "rmse", "root_mean_squared_error": "rmse", "mae": "mae",
       "mean_absolute_error": "mae", "r2": "r2", "r2_score": "r2"}.get(primary.lower())
if key is None:
    raise RuntimeError(f"Unsupported primary metric: {primary!r}")
lower_better = key in ("rmse", "mae")
best_name = min(scores, key=lambda n: scores[n][f"{key}_macro"]) if lower_better else max(scores, key=lambda n: scores[n][f"{key}_macro"])
best_score = scores[best_name][f"{key}_macro"]

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics by candidate, target and macro average.")
mlca_runtime.record_metric(primary, best_score, "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")

print(f"features={len(feature_cols)} val_rows={len(val)} candidates={len(candidates)}")
print(f"best_model={best_name} {primary}={best_score:.4g}")
