import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, recall_score
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
task = protocol["task"]
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
primary = task["primary_metric"]

train = pd.read_parquet(inp["train"])
test = pd.read_parquet(inp["test"])

def key_like(s):
    if len(s) <= 1 or not s.is_unique:
        return False
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_integer_dtype(s))

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feature_cols = [c for c in train.columns if c not in drop and not key_like(train[c])]

def read_any(path):
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)

def get_metric(df, name):
    for mc in ["metric", "metric_name", "name"]:
        if mc in df.columns:
            vc = next((c for c in ["value", "score", "mean"] if c in df.columns), None)
            if vc:
                hit = df[df[mc].astype(str).str.lower() == name.lower()]
                if len(hit):
                    return float(hit[vc].iloc[0])
    if name in df.columns:
        return float(df[name].iloc[0])
    raise KeyError(name)

val_keys = [
    "10_validate_score_validation_7W7VKTVE/validation_metrics.parquet",
    "11_validate_score_validation_5PKKGYTK/validation_metrics.parquet",
]
vals = [read_any(inp[k]) for k in val_keys]
aucs = [get_metric(v, "roc_auc") for v in vals]
i = int(np.argmax(aucs))
model_key = ["08_model_train_logistic_Y7873DWF/model.pkl",
             "09_model_train_alternative_F9H6TZDH/alternative_model.pkl"][i]
manifest_key = ["08_model_train_logistic_Y7873DWF/manifest.json",
                "09_model_train_alternative_F9H6TZDH/manifest.json"][i]

with open(inp[model_key], "rb") as f:
    model = pickle.load(f)
with open(inp["06_preprocess_fit_preprocessing_STWKFGVD/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["07_features_select_features_N6F5BDYX/selector.pkl"], "rb") as f:
    sel = pickle.load(f)
manifest = json.loads(inp[manifest_key].read_text())

def find_thr(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if "threshold" in str(k).lower() and isinstance(v, (int, float)):
                return float(v)
            r = find_thr(v)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find_thr(v)
            if r is not None:
                return r
    return None

thr = find_thr(manifest)
thr = find_thr(vals[i]) if thr is None else thr
if thr is None:
    vdf = vals[i]
    if "metric" in vdf.columns and "value" in vdf.columns:
        hit = vdf[vdf["metric"].astype(str).str.lower() == "threshold"]
        if len(hit):
            thr = float(hit["value"].iloc[0])
thr = 0.5 if thr is None else float(thr)

classes = list(getattr(model, "classes_", []))
if not classes and hasattr(model, "steps"):
    classes = list(getattr(model.steps[-1][1], "classes_", []))
if not classes:
    raise RuntimeError("model classes missing")

def cols_for(obj):
    names = getattr(obj, "feature_names_in_", None)
    return list(names) if names is not None else feature_cols

def wrap(obj, X, idx):
    if hasattr(X, "toarray"):
        X = X.toarray()
    if isinstance(X, pd.DataFrame):
        return X
    names = getattr(obj, "feature_names_in_", None)
    if names is None and hasattr(obj, "get_feature_names_out"):
        try:
            names = obj.get_feature_names_out()
        except Exception:
            names = None
    if names is not None and len(names) == X.shape[1]:
        return pd.DataFrame(X, columns=list(names), index=idx)
    return X

def transform(obj, X, idx):
    if hasattr(X, "toarray"):
        X = X.toarray()
    names = getattr(obj, "feature_names_in_", None)
    if not isinstance(X, pd.DataFrame) and names is not None and len(names) == X.shape[1]:
        X = pd.DataFrame(X, columns=list(names), index=idx)
    if isinstance(X, pd.DataFrame):
        X = X[cols_for(obj)]
    return wrap(obj, obj.transform(X), idx)

def model_input(m, X, idx):
    if hasattr(X, "toarray"):
        X = X.toarray()
    names = getattr(m, "feature_names_in_", None)
    if names is not None:
        names = list(names)
        if isinstance(X, pd.DataFrame):
            return X[names]
        if len(names) == X.shape[1]:
            return pd.DataFrame(X, columns=names, index=idx)
    return X

def predict_seq(kind, df):
    idx = df.index
    if kind == "model":
        return model.predict_proba(model_input(model, df, idx))
    if kind == "sel_model":
        Z = transform(sel, df, idx)
        return model.predict_proba(model_input(model, Z, idx))
    if kind == "pre_sel_model":
        Z = transform(pre, df, idx)
        Z = transform(sel, Z, idx)
        return model.predict_proba(model_input(model, Z, idx))
    if kind == "pre_model":
        Z = transform(pre, df, idx)
        return model.predict_proba(model_input(model, Z, idx))

sample = train.iloc[:min(20, len(train))]
proba = None
kind = None
for k in ["model", "sel_model", "pre_sel_model", "pre_model"]:
    try:
        q = np.asarray(predict_seq(k, sample))
        if q.ndim == 2 and q.shape[0] == len(sample) and q.shape[1] == len(classes):
            proba, kind = q, k
            break
    except Exception:
        pass
if proba is None:
    raise RuntimeError("No compatible saved prediction pipeline")

proba = np.asarray(predict_seq(kind, test))
target = targets[0]
pos = task.get("positive_label", task.get("positive_class", classes[1]))
if pos not in classes:
    pos = classes[1]
neg = [c for c in classes if c != pos][0]
jpos = classes.index(pos)
y = test[target].to_numpy()
p = proba[:, jpos]
pred = np.where(p >= thr, pos, neg)

def metrics_from(yv, pv, predv):
    yb = (yv == pos).astype(int)
    pb = (predv == pos)
    return {
        "roc_auc": roc_auc_score(yb, pv),
        "pr_auc": average_precision_score(yb, pv),
        "balanced_accuracy": balanced_accuracy_score(yb, pb),
        "brier_score": brier_score_loss(yb, pv),
        "sensitivity": recall_score(yb, pb, pos_label=1, zero_division=0),
        "specificity": recall_score(yb, pb, pos_label=0, zero_division=0),
    }

point = metrics_from(y, p, pred)
rng = np.random.default_rng(42)
boot = {k: [] for k in point}
n = len(y)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    if len(np.unique(y[idx])) < 2:
        for k in boot:
            boot[k].append(np.nan)
        continue
    try:
        bm = metrics_from(y[idx], p[idx], pred[idx])
        for k in boot:
            boot[k].append(bm[k])
    except Exception:
        for k in boot:
            boot[k].append(np.nan)

ci = {}
for k, arr in boot.items():
    arr = np.asarray(arr, dtype=float)
    if np.isfinite(arr).any():
        ci[k] = (float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5)))
    else:
        ci[k] = (np.nan, np.nan)

rows = []
for k in ["roc_auc", "pr_auc", "balanced_accuracy", "brier_score", "sensitivity", "specificity"]:
    rows.append({"metric": k, "value": point[k], "ci_low": ci[k][0], "ci_high": ci[k][1]})
metrics_df = pd.DataFrame(rows)
for c in ["value", "ci_low", "ci_high"]:
    metrics_df[c] = metrics_df[c].astype(float).round(4)
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics with 95% bootstrap CIs")

for k in point:
    mlca_runtime.record_metric(k, float(point[k]), "test", ci[k])
if "roc_auc" in point:
    mlca_runtime.record_fact_candidate("test_roc_auc", float(point["roc_auc"]), source="test_metrics.csv")

pred_df = pd.DataFrame(index=test.index)
pred_df["row_id"] = test.index
pred_df["partition"] = "test"
pred_df[f"y_true__{target}"] = y
pred_df[f"y_pred__{target}"] = pred
for j, c in enumerate(classes):
    pred_df[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred_df.reset_index(drop=True), "test_predictions.parquet")

print(f"test rows={len(test)} selected={kind} threshold={thr:.4f}")
print(f"{primary}={point[primary]:.4f} CI=({ci[primary][0]:.4f},{ci[primary][1]:.4f})")
