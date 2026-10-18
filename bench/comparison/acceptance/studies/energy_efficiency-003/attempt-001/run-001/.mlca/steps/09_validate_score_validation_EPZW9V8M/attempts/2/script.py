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
task = need(protocol, "task")
primary = need(task, "primary_metric")
group_col = protocol.get("group_column")

protected = set(targets) | set(excluded) | ({group_col} if group_col else set())
feature_cols = [c for c in train.columns if c not in protected]

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
    return Pipeline([("preprocessor", pre), ("selector", sel), ("model", obj)])

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

def pred_for(predictor):
    fni = getattr(predictor, "feature_names_in_", None)
    if fni is not None:
        fni = list(fni)
        bad = [c for c in fni if c in protected]
        if bad:
            raise RuntimeError(f"pipeline expects protected columns: {bad}")
        missing = [c for c in fni if c not in val.columns]
        if missing:
            raise RuntimeError(f"pipeline expects missing columns: {missing}")
        X = val[fni].copy()
    else:
        X = val[feature_cols].copy()
    p = np.asarray(predictor.predict(X))
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    if p.shape[1] != len(targets):
        raise RuntimeError(f"prediction has {p.shape[1]} columns, expected {len(targets)}")
    return p

rows, scores = [], {}
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
picker = min if lower_better else max
best_name = picker(scores, key=lambda n: scores[n][f"{key}_macro"])
best_score = scores[best_name][f"{key}_macro"]

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics by candidate, target, and macro average.")
mlca_runtime.record_metric(primary, best_score, "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")

print(f"features={len(feature_cols)} val_rows={len(val)} candidates={len(candidates)}")
print(f"best_model={best_name} {primary}={best_score:.4g}")
