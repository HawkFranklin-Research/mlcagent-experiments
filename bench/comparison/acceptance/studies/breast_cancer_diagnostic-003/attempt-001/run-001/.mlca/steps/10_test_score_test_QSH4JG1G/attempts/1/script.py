import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score)

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
out = mlca_runtime.output_dir()
protocol = json.loads(paths["_protocol"].read_text())
try:
    task = protocol["task"]
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    group_col = protocol["group_column"]
    primary = task["primary_metric"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available top-level keys: {list(protocol.keys())}")

test = pd.read_parquet(paths["test"])
train = pd.read_parquet(paths["train"])
model = pickle.loads(paths["07_model_train_logistic_QMMJ5EQH/model.pkl"].read_bytes())

target = targets[0]
feat_cols = [c for c in train.columns if c not in set(targets) | set(excluded) | {group_col}]
key_like = [c for c in feat_cols
            if (pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]))
            and train[c].nunique() == len(train)]
feat_cols = [c for c in feat_cols if c not in key_like]
X_test = test[feat_cols].copy()
y_test = test[target]

classes = model.classes_
if task.get("type") == "binary" or len(classes) == 2:
    proba = model.predict_proba(X_test)
    pos_label = classes[1]
    p = proba[:, list(classes).index(pos_label)]
    y_bin = (y_test == pos_label).astype(int)
    val = pd.read_parquet(paths["val"])
    X_val = val[feat_cols]
    pv = model.predict_proba(X_val)[:, list(classes).index(pos_label)]
    yv_bin = (val[target] == pos_label).astype(int)
    thr = float(np.quantile(pv, 1 - yv_bin.mean()))
    metrics = {"roc_auc": roc_auc_score(y_bin, p),
               "pr_auc": average_precision_score(y_bin, p),
               "balanced_accuracy": balanced_accuracy_score(y_bin, (p >= thr).astype(int)),
               "brier": brier_score_loss(y_bin, p)}
    tn, fp, fn, tp = confusion_matrix(y_bin, (p >= thr).astype(int)).ravel()
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) else np.nan
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
    rng = np.random.default_rng(42)
    idx_all = np.arange(len(y_bin))
    boot = {k: [] for k in metrics}
    for _ in range(1000):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        if len(np.unique(y_bin.iloc[idx])) < 2:
            continue
        yb = y_bin.iloc[idx]; pb = p[idx]
        boot["roc_auc"].append(roc_auc_score(yb, pb))
        boot["pr_auc"].append(average_precision_score(yb, pb))
        predb = (pb >= thr).astype(int)
        boot["balanced_accuracy"].append(balanced_accuracy_score(yb, predb))
        boot["brier"].append(brier_score_loss(yb, pb))
        tnb, fpb, fnb, tpb = confusion_matrix(yb, predb, labels=[0, 1]).ravel()
        boot["sensitivity"].append(tpb / (tpb + fnb) if (tpb + fnb) else np.nan)
        boot["specificity"].append(tnb / (tnb + fpb) if (tnb + fpb) else np.nan)
    rows = []
    for k, v in metrics.items():
        arr = np.array(boot[k], dtype=float)
        arr = arr[~np.isnan(arr)]
        lo, hi = (np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan))
        rows.append({"metric": k, "value": v, "ci_low": lo, "ci_high": hi})
    df = pd.DataFrame(rows)
else:
    proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    r = roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=classes)
    f1 = f1_score(y_test, y_pred, average="macro")
    rng = np.random.default_rng(42)
    idx_all = np.arange(len(y_test))
    br, bf = [], []
    for _ in range(1000):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        if len(np.unique(y_test.iloc[idx])) < 2:
            continue
        br.append(roc_auc_score(y_test.iloc[idx], proba[idx], multi_class="ovr",
                                average="macro", labels=classes))
        bf.append(f1_score(y_test.iloc[idx], y_pred[idx], average="macro"))
    rows = []
    for k, v, arr in [("roc_auc_ovr_macro", r, np.array(br)), ("f1_macro", f1, np.array(bf))]:
        lo, hi = (np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan))
        rows.append({"metric": k, "value": v, "ci_low": lo, "ci_high": hi})
    df = pd.DataFrame(rows)

df = df.round(4)
mlca_runtime.save_table(df, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")
prim_row = df[df["metric"] == primary]
if len(prim_row):
    pr = prim_row.iloc[0]
    mlca_runtime.record_metric(primary, float(pr["value"]), "test",
                               ci=(float(pr["ci_low"]), float(pr["ci_high"])))
if primary == "roc_auc":
    pr = df[df["metric"] == "roc_auc"].iloc[0]
    mlca_runtime.record_fact_candidate("test_roc_auc", float(pr["value"]),
                                       source="test_metrics.csv")
print(f"test rows={len(test)} features={len(feat_cols)}")
print(f"{primary}={float(df[df['metric']==primary]['value'].iloc[0]):.4f}")
