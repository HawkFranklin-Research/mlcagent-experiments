import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing {k}; available top-level keys: {list(protocol.keys())}")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"] or [])
group = protocol["group_column"]
pm = protocol["task"]["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def key_like(s):
    n = s.nunique(dropna=True)
    if n <= 1 or n != len(s):
        return False
    return pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or pd.api.types.is_integer_dtype(s.dtype)

base_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
feature_cols = [c for c in base_cols if not key_like(train[c])]
for c in feature_cols:
    if c not in val.columns:
        raise KeyError(f"Missing feature column in val: {c}")
X_val = val[feature_cols].copy()
y_val = val[targets].copy()

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load(inp["05_preprocess_fit_preprocessing_GTAJ2HZB/preprocessor.pkl"])
sel = load(inp["06_features_retain_features_XC2ENAE2/selector_all.pkl"])
models = {
    "baseline": load(inp["07_model_train_baseline_25GK6Z71/model.pkl"]),
    "alternative": load(inp["08_model_train_alternative_VF3B36TN/alternative_model.pkl"]),
}

def raw_ok(obj):
    try:
        names = list(obj.feature_names_in_)
    except Exception:
        names = None
    if names is not None and len(names) > 0:
        return set(names).issubset(set(feature_cols))
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            try:
                sn = list(step.feature_names_in_)
            except Exception:
                continue
            if len(sn) > 0 and set(sn).issubset(set(feature_cols)):
                return True
    return False

def has_pre(obj):
    if not hasattr(obj, "named_steps"):
        return False
    for name, step in obj.named_steps.items():
        if "preprocess" in name.lower() or isinstance(step, ColumnTransformer):
            return True
    return False

def frontend():
    if sel is not None and has_pre(sel):
        return sel
    if sel is not None:
        if raw_ok(sel):
            return sel
        if raw_ok(pre):
            return Pipeline([("preprocessor", pre), ("selector", sel)])
    if raw_ok(pre):
        return pre
    return None

def make_pipe(obj):
    if has_pre(obj) or raw_ok(obj):
        return obj
    fe = frontend()
    if fe is None:
        raise RuntimeError("No saved frontend accepts the raw feature DataFrame")
    if hasattr(obj, "named_steps") and any("selector" in n.lower() for n in obj.named_steps) and raw_ok(pre):
        return Pipeline([("preprocessor", pre), ("model", obj)])
    return Pipeline([("frontend", fe), ("model", obj)])

rows = []
scores = {}
for name, obj in models.items():
    pipe = make_pipe(obj)
    pred = np.asarray(pipe.predict(X_val))
    if len(targets) == 1:
        yt_all = y_val[targets[0]].to_numpy()
        yp_all = pred.ravel()
    else:
        yt_all = y_val[targets].to_numpy()
        yp_all = pred
    md = {}
    for i, t in enumerate(targets):
        yt, yp = (yt_all, yp_all) if len(targets) == 1 else (yt_all[:, i], yp_all[:, i])
        md[f"rmse_{t}"] = float(np.sqrt(mean_squared_error(yt, yp)))
        md[f"mae_{t}"] = float(mean_absolute_error(yt, yp))
        md[f"r2_{t}"] = float(r2_score(yt, yp))
    md["rmse_macro"] = float(np.mean([md[f"rmse_{t}"] for t in targets]))
    md["mae_macro"] = float(np.mean([md[f"mae_{t}"] for t in targets]))
    md["r2_macro"] = float(np.mean([md[f"r2_{t}"] for t in targets]))
    scores[name] = md
    for k, v in md.items():
        rows.append({"model": name, "metric": k, "value": v})

pml = pm.lower()
lower_better = pml in {"rmse", "mae", "mse", "mape", "log_loss", "mean_squared_error", "mean_absolute_error"}
higher_better = pml in {"r2", "r2_score", "accuracy", "f1", "f1_score", "roc_auc", "precision", "recall", "explained_variance"}
key = f"{pml}_macro" if f"{pml}_macro" in next(iter(scores.values())) else pml
if key not in next(iter(scores.values())):
    key = "rmse_macro"
if lower_better:
    best_name = min(scores, key=lambda n: scores[n][key])
elif higher_better:
    best_name = max(scores, key=lambda n: scores[n][key])
else:
    best_name = min(scores, key=lambda n: scores[n]["rmse_macro"])
best_score = float(scores[best_name][key])

vdf = pd.DataFrame(rows)
vdf["value"] = vdf["value"].map(lambda x: float(f"{x:.4g}") if pd.notna(x) else x)
mlca_runtime.save_table(vdf, "validation_metrics.csv", caption="Validation metrics by candidate")
mlca_runtime.record_metric(pm, best_score, "val")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
print(f"val rows={len(val)} features={len(feature_cols)}")
print(f"best_model={best_name} {pm}={best_score:.4g}")
