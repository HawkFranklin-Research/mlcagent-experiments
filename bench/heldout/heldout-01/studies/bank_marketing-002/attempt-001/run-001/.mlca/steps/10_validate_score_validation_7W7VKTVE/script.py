import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss, roc_curve)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
task = protocol["task"]
primary = task["primary_metric"]
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
print(f"train={train.shape} val={val.shape} primary={primary} target={target}")

# Feature columns (RL-0007)
excl = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in excl]
for c in list(feature_cols):
    s = train[c]
    if s.dtype == object or str(s.dtype).startswith(("string", "category")):
        if s.nunique(dropna=True) == len(s):
            feature_cols.remove(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=True) == len(s):
            feature_cols.remove(c)

y_val = val[target]

# Candidates
cand_paths = {
    "logistic": inputs["08_model_train_logistic_Y7873DWF/model.pkl"],
    "alternative": inputs["09_model_train_alternative_F9H6TZDH/alternative_model.pkl"],
}

# Determine optimization direction for the declared primary metric
LOWER_IS_BETTER = {"brier", "log_loss", "rmse"}

records = []
model_scores = {}
for name, path in cand_paths.items():
    with open(path, "rb") as f:
        mdl = pickle.load(f)
    cols = list(getattr(mdl, "feature_names_in_", feature_cols))
    Xv = val[cols]
    proba = mdl.predict_proba(Xv)
    classes = list(mdl.classes_)
    pos_label = classes[1]  # RL-0017: no declared pos label -> classes_[1]
    p = proba[:, 1]
    y_bin = (y_val.values == pos_label).astype(int)

    auc = float(roc_auc_score(y_bin, p))
    ap = float(average_precision_score(y_bin, p))
    brier = float(brier_score_loss(y_bin, p))

    fpr, tpr, thr = roc_curve(y_bin, p)
    m = np.isfinite(thr)
    fpr, tpr, thr = fpr[m], tpr[m], thr[m]
    j = tpr - fpr
    best_thr = float(thr[int(np.argmax(j))])
    yhat = (p >= best_thr).astype(int)
    bal = float(balanced_accuracy_score(y_bin, yhat))

    records.append({"model": name, "metric": primary, "value": auc, "threshold": np.nan})
    records.append({"model": name, "metric": "pr_auc", "value": ap, "threshold": np.nan})
    records.append({"model": name, "metric": "balanced_accuracy", "value": bal, "threshold": best_thr})
    records.append({"model": name, "metric": "brier", "value": brier, "threshold": np.nan})

    model_scores[name] = auc
    mlca_runtime.record_metric(f"{name}_{primary}", auc, "val")
    print(f"{name}: {primary}={auc:.4f} threshold={best_thr:.4f}")

df = pd.DataFrame(records)
for c in ["value", "threshold"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype(float).round(4)
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics by candidate model")

scores = {k: v for k, v in model_scores.items()}
best_name = min(scores, key=scores.get) if primary in LOWER_IS_BETTER else max(scores, key=scores.get)
best_val = float(scores[best_name])
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_val, source="validation_metrics.csv")
print(f"best={best_name} {primary}={best_val:.4f}")
