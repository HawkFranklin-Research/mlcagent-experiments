import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             balanced_accuracy_score, roc_curve, f1_score)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
task = proto["task"]
primary_metric = task["primary_metric"]
targets = proto["targets"]
excluded = proto.get("excluded_columns", [])
group = proto.get("group_column")
target_col = targets[0]
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def load_pkl(k):
    with inp[k].open("rb") as f:
        return pickle.load(f)

pre = load_pkl("05_preprocess_fit_preprocessing_6FVQ17HJ/preprocessor.pkl")
sel = load_pkl("06_features_select_features_R6V6W10Y/selector.pkl")
models = {
    "logistic": load_pkl("07_model_train_logistic_XYKSK7V0/model.pkl"),
    "alternative": load_pkl("08_model_train_alternative_6QT8CRT5/alternative_model.pkl"),
}

exclude_set = set(targets) | set(excluded) | ({group} if group else set())
cands = []
for obj in [pre, sel] + list(models.values()):
    if obj is not None and hasattr(obj, "feature_names_in_"):
        cols = [c for c in obj.feature_names_in_ if c not in exclude_set]
        if cols and all(c in train.columns for c in cols):
            cands.append(cols)
raw_cols = max(cands, key=len) if cands else [
    c for c in train.columns if c not in exclude_set
]
X_val = val[raw_cols].copy()
y_val = val[target_col]

def get_classes(m):
    if hasattr(m, "classes_"):
        return np.asarray(m.classes_)
    if hasattr(m, "named_steps"):
        return np.asarray(list(m.named_steps.values())[-1].classes_)
    raise ValueError("No classes_ found on model")

def predict_proba_candidate(m, raw_df):
    if hasattr(m, "named_steps"):
        cols = getattr(m, "feature_names_in_", None)
        if cols is not None and all(c in raw_df.columns for c in cols):
            return m.predict_proba(raw_df[list(cols)])
        if pre is not None and hasattr(pre, "feature_names_in_"):
            return m.predict_proba(pre.transform(raw_df[list(pre.feature_names_in_)]))
        return m.predict_proba(raw_df)
    if sel is not None:
        if hasattr(sel, "named_steps"):
            scols = getattr(sel, "feature_names_in_", None)
            if scols is not None and all(c in raw_df.columns for c in scols):
                Xt = sel.transform(raw_df[list(scols)])
            elif pre is not None and hasattr(pre, "feature_names_in_"):
                Xt = sel.transform(pre.transform(raw_df[list(pre.feature_names_in_)]))
            else:
                Xt = sel.transform(raw_df)
        elif pre is not None and hasattr(pre, "feature_names_in_"):
            Xt = sel.transform(pre.transform(raw_df[list(pre.feature_names_in_)]))
        else:
            scols = getattr(sel, "feature_names_in_", None)
            Xt = sel.transform(raw_df[list(scols)] if scols else raw_df)
    elif pre is not None and hasattr(pre, "feature_names_in_"):
        Xt = pre.transform(raw_df[list(pre.feature_names_in_)])
    else:
        Xt = raw_df
    if hasattr(m, "feature_names_in_") and hasattr(Xt, "columns"):
        mcols = [c for c in m.feature_names_in_ if c in Xt.columns]
        if len(mcols) == len(m.feature_names_in_):
            Xt = Xt[mcols]
    return m.predict_proba(Xt)

rows, scores = [], {}
first_cls = get_classes(next(iter(models.values())))
is_binary = len(first_cls) == 2
for name, m in models.items():
    proba = predict_proba_candidate(m, X_val)
    classes = get_classes(m)
    if is_binary:
        pos_label = task.get("positive_label")
        if pos_label is None:
            pos_label = classes[1]
        j = int(np.where(classes == pos_label)[0][0])
        p = proba[:, j]
        yb = (y_val == pos_label).astype(int)
        roc = roc_auc_score(yb, p)
        pr = average_precision_score(yb, p)
        brier = brier_score_loss(yb, p)
        fpr, tpr, thr = roc_curve(yb, p)
        finite = np.isfinite(thr)
        if finite.any():
            jj = int(np.argmax(tpr[finite] - fpr[finite]))
            best_thr = float(thr[finite][jj])
        else:
            best_thr = 0.5
        bal = balanced_accuracy_score(yb, (p >= best_thr).astype(int))
        rows += [(name, "roc_auc", roc, np.nan),
                 (name, "pr_auc", pr, np.nan),
                 (name, "balanced_accuracy", bal, best_thr),
                 (name, "brier_score", brier, np.nan)]
        scores[name] = dict(roc_auc=roc, pr_auc=pr,
                            balanced_accuracy=bal, brier_score=brier)
    else:
        present = [c for c in classes if c in set(y_val)]
        if len(present) == len(classes):
            auc = roc_auc_score(y_val, proba, multi_class="ovr",
                                average="macro", labels=classes)
        else:
            vals = [roc_auc_score((y_val == c).astype(int), proba[:, j])
                    for j, c in enumerate(classes) if c in present]
            auc = float(np.mean(vals)) if vals else np.nan
            print("omitted classes", [c for c in classes if c not in present])
        pred = classes[np.argmax(proba, axis=1)]
        f1 = f1_score(y_val, pred, average="macro")
        rows += [(name, "roc_auc_ovr_macro", auc, np.nan),
                 (name, "f1_macro", f1, np.nan)]
        scores[name] = dict(roc_auc_ovr_macro=auc, f1_macro=f1)

if is_binary:
    primary_scores = {k: v.get(primary_metric, v["roc_auc"]) for k, v in scores.items()}
    best_roc = lambda k: scores[k]["roc_auc"]
else:
    pk = "f1_macro" if primary_metric == "f1" else "roc_auc_ovr_macro"
    primary_scores = {k: v[pk] for k, v in scores.items()}
    best_roc = lambda k: scores[k]["roc_auc_ovr_macro"]

lower = any(x in primary_metric.lower() for x in
            ["brier", "loss", "error", "mse", "rmse", "mae"])
best_name = (min(primary_scores, key=primary_scores.get) if lower
             else max(primary_scores, key=primary_scores.get))
best_score = float(primary_scores[best_name])

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
def sig4(x):
    return np.nan if pd.isna(x) else float(f"{x:.4g}")
metrics_df["value"] = metrics_df["value"].map(sig4)
metrics_df["threshold"] = metrics_df["threshold"].map(sig4)
mlca_runtime.save_table(metrics_df, "validation_metrics.csv")

mlca_runtime.record_metric(primary_metric, best_score, "val")
mlca_runtime.record_fact_candidate("best_model", best_name,
                                   source="validation_metrics.csv")
if primary_metric == "roc_auc":
    mlca_runtime.record_fact_candidate("best_val_roc_auc",
                                       float(best_roc(best_name)),
                                       source="validation_metrics.csv")

print(f"train {train.shape}, val {val.shape}")
print(f"raw features {len(raw_cols)}")
print(f"best {best_name}: {primary_metric}={best_score:.4f}")
print("saved validation_metrics.csv")
