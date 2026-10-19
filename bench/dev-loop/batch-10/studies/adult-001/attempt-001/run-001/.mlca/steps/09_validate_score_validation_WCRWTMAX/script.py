import mlca_runtime
import pickle, json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, roc_curve,
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
task = protocol["task"]
primary_name = task["primary_metric"]
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group_col = protocol.get("group_column")
n_classes = int(task.get("n_classes") or 2)
is_multiclass = n_classes > 2

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

drop = set(targets + excluded + ([group_col] if group_col else []))
feature_cols = [c for c in train.columns if c not in drop]
key_like = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if len(s) > 1 and s.nunique(dropna=True) == len(s):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        if len(s) > 1 and s.nunique(dropna=True) == len(s) and c.lower().endswith("id"):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

def load_pkl(k):
    with open(inp[k], "rb") as f:
        return pickle.load(f)

pre = load_pkl("05_preprocess_fit_preprocessing_F9B0NJ26/preprocessor.pkl")
sel = load_pkl("06_features_select_features_AA6KHSXZ/selector.pkl")
model1 = load_pkl("07_model_train_logistic_23SCN2HD/model.pkl")
model2 = load_pkl("08_model_train_alternative_9VT6FVM1/alternative_model.pkl")

def names(obj):
    if hasattr(obj, "feature_names_in_") and obj.feature_names_in_ is not None:
        return [str(x) for x in obj.feature_names_in_]
    return None

raw_set = set(feature_cols)
sn = names(sel) if sel is not None else None
pn = names(pre) if pre is not None else None
front = None
if sel is not None and sn is not None and set(sn).issubset(raw_set):
    front = sel
elif sel is not None and pre is not None and pn is not None and set(pn).issubset(raw_set):
    if sn is None or not set(sn).issubset(raw_set):
        front = Pipeline([("preprocessor", pre), ("selector", sel)])
elif sel is not None and sn is None:
    front = sel
elif pre is not None and (pn is None or set(pn).issubset(raw_set)):
    front = pre

def candidate_for(m):
    mn = names(m)
    if isinstance(m, Pipeline):
        if mn is None or set(mn).issubset(raw_set):
            return m
    elif mn is not None and set(mn).issubset(raw_set):
        return m
    if front is not None:
        return Pipeline([("frontend", front), ("model", m)])
    return m

def get_X(model, df):
    mn = names(model)
    if mn is not None:
        miss = [c for c in mn if c not in df.columns]
        if miss:
            raise ValueError(f"missing model input columns: {miss}")
        return df[mn]
    return df[feature_cols]

def evaluate(name, model_obj):
    model = candidate_for(model_obj)
    X = get_X(model, val)
    proba = np.asarray(model.predict_proba(X), dtype=float)
    classes = list(getattr(model, "classes_", [0, 1]))
    y = val[targets[0]].astype(object).to_numpy()
    yu = list(pd.unique(pd.Series(y).dropna()))
    pos = classes[-1] if len(classes) >= 2 else classes[0]
    if pos not in yu:
        for u in yu:
            if str(u) == str(pos):
                pos = u
                break
    if pos not in yu and len(yu) >= 2:
        pos = yu[-1]
    if pos in classes:
        p = proba[:, classes.index(pos)]
    else:
        p = proba[:, -1]
    p = np.asarray(p, dtype=float)
    y_bin = np.array([1 if v == pos else 0 for v in y], dtype=int)
    mask = np.isfinite(p) & pd.notna(pd.Series(y)).to_numpy()
    p, y_bin = p[mask], y_bin[mask]
    if len(np.unique(y_bin)) < 2:
        raise ValueError(f"{name}: validation y degenerate for pos={pos}")
    auc = float(roc_auc_score(y_bin, p))
    ap = float(average_precision_score(y_bin, p))
    fpr, tpr, thr = roc_curve(y_bin, p)
    j = tpr - fpr
    ft = np.isfinite(thr)
    idx = np.where(ft)[0] if ft.any() else np.arange(len(thr))
    threshold = float(thr[idx[np.argmax(j[idx])]]) if len(idx) else 0.5
    y_pred = (p >= threshold).astype(int)
    ba = float(balanced_accuracy_score(y_bin, y_pred))
    brier = float(brier_score_loss(y_bin, p))
    return {"model": name, "roc_auc": auc, "pr_auc": ap,
            "balanced_accuracy": ba, "brier_score": brier,
            "threshold": threshold, "pos": str(pos)}

results = [evaluate("logistic", model1), evaluate("alternative", model2)]

key_map = {"roc_auc": "roc_auc", "pr_auc": "pr_auc",
           "balanced_accuracy": "balanced_accuracy",
           "brier_score": "brier_score", "brier": "brier_score",
           "average_precision": "pr_auc"}
sel_key = key_map.get(primary_name, "roc_auc")
higher_better = primary_name not in {"brier_score", "brier", "log_loss", "rmse", "mae"}

rows = []
metric_names = ["roc_auc", "pr_auc", "balanced_accuracy", "brier_score"]
for r in results:
    for m in metric_names:
        thr_val = np.nan if is_multiclass else r["threshold"]
        rows.append({"model": r["model"], "metric": m,
                     "value": r[m], "threshold": thr_val})
    mlca_runtime.record_metric(primary_name, float(r[sel_key]), split="val")

metrics_df = pd.DataFrame(rows)
metrics_df["value"] = metrics_df["value"].apply(lambda x: float(f"{x:.4g}"))
metrics_df["threshold"] = metrics_df["threshold"].apply(
    lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)
mlca_runtime.save_table(metrics_df, "validation_metrics.csv",
                        caption="Validation metrics per candidate model")

best = (max if higher_better else min)(results, key=lambda r: r[sel_key])
mlca_runtime.record_fact_candidate("best_model", best["model"], source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best["roc_auc"]), source="validation_metrics.csv")

print("train", train.shape, "val", val.shape)
print("features", len(feature_cols), "primary", primary_name)
print("direction", "higher" if higher_better else "lower")
print("best", best["model"], float(best[sel_key]))
