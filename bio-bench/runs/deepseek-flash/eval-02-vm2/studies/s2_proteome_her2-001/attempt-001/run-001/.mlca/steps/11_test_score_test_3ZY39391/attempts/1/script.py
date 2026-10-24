import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, recall_score, f1_score
)
from sklearn.pipeline import Pipeline
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing key {k!r}; available top-level keys: {list(d)}")
    return d[k]

targets = req(protocol, "targets")
excluded = set(req(protocol, "excluded_columns"))
group = protocol.get("group_column")
task = req(protocol, "task")
primary_metric = req(task, "primary_metric")
task_type = task.get("type", "binary" if task.get("n_classes") == 2 else "multiclass")
positive_label = task.get("positive_label") or task.get("positive_class") or protocol.get("positive_label")

train = pd.read_parquet(inp["train"])
all_cols = list(train.columns)
feat_cols = [c for c in all_cols if c not in set(targets) and c not in excluded and c != group]
n = len(train)
key_like = []
for c in feat_cols:
    s = train[c]
    if s.nunique(dropna=False) == n and (
        pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)
    ):
        key_like.append(c)
feat_cols = [c for c in feat_cols if c not in key_like]

def load_pkl(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("06_preprocess_fit_preprocessing_R9QMSZCE/preprocessor.pkl")
sel = load_pkl("07_features_select_features_C8HA72DE/selector.pkl")
model = load_pkl("latest_model")
test = pd.read_parquet(inp["test"])

# If the saved model expects transformed names absent from test, assemble the saved chain.
cols = list(getattr(model, "feature_names_in_", []))
if cols and not all(c in test.columns for c in cols):
    sel_cols = list(getattr(sel, "feature_names_in_", []))
    if sel_cols and all(c in test.columns for c in sel_cols):
        model = Pipeline([("selector", sel), ("model", model)])
    else:
        model = Pipeline([("preprocessor", pre), ("selector", sel), ("model", model)])

def input_frame(m, df):
    cols = list(getattr(m, "feature_names_in_", []))
    if cols and all(c in df.columns for c in cols):
        return df[cols]
    return df[feat_cols]

X_test = input_frame(model, test)
y_pred = model.predict(X_test)
proba = model.predict_proba(X_test)
classes = np.asarray(model.classes_)

threshold = 0.5
try:
    vm = pd.read_csv(inp["10_validate_score_validation_0SZC4097/validation_metrics.csv"])
    for c in ("threshold", "threshold_value"):
        if c in vm.columns:
            vals = pd.to_numeric(vm[c], errors="coerce").dropna()
            if len(vals):
                threshold = float(vals.mode().iloc[0])
                break
except Exception:
    pass
if hasattr(model, "threshold_"):
    threshold = float(model.threshold_)

target = targets[0]
y_true = test[target].to_numpy()

if task_type == "binary":
    if positive_label is None or positive_label not in classes:
        positive_label = classes[1]
    pos_idx = int(np.where(classes == positive_label)[0][0])
    y_bin = (y_true == positive_label).astype(int)
    p_pos = proba[:, pos_idx]

    def metrics(yb, p, yp):
        ypb = (p >= threshold).astype(int)
        out = {}
        out["roc_auc"] = roc_auc_score(yb, p) if len(np.unique(yb)) == 2 else np.nan
        out["pr_auc"] = average_precision_score(yb, p) if len(np.unique(yb)) == 2 else np.nan
        out["balanced_accuracy"] = balanced_accuracy_score(yb, ypb) if len(np.unique(yb)) == 2 else np.nan
        out["brier_score"] = brier_score_loss(yb, p)
        out["sensitivity"] = recall_score(yb, ypb, pos_label=1, zero_division=0) if yb.sum() > 0 else np.nan
        out["specificity"] = recall_score(yb, ypb, pos_label=0, zero_division=0) if (len(yb) - yb.sum()) > 0 else np.nan
        return out

    point = metrics(y_bin, p_pos, y_pred)
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, len(y_bin), len(y_bin))
        boot.append(metrics(y_bin[idx], p_pos[idx], y_pred[idx]))
else:
    present = [c for c in classes if np.any(y_true == c)]
    omitted = [c for c in classes if c not in present]
    if omitted:
        print(f"omitted classes in test: {omitted}")

    def metrics(y, pr):
        yp = classes[np.argmax(pr, axis=1)]
        aucs = []
        for c in classes:
            if not np.any(y == c):
                continue
            j = int(np.where(classes == c)[0][0])
            yb = (y == c).astype(int)
            if 0 < yb.sum() < len(yb):
                aucs.append(roc_auc_score(yb, pr[:, j]))
        roc = float(np.mean(aucs)) if aucs else np.nan
        f1 = f1_score(y, yp, average="macro", labels=classes, zero_division=0)
        return {"roc_auc_ovr_macro": roc, "f1_macro": f1}

    point = metrics(y_true, proba)
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, len(y_true), len(y_true))
        boot.append(metrics(y_true[idx], proba[idx]))

rows = []
for k, v in point.items():
    vals = [b[k] for b in boot if k in b and np.isfinite(b[k])]
    lo = float(np.percentile(vals, 2.5)) if len(vals) >= 2 else np.nan
    hi = float(np.percentile(vals, 97.5)) if len(vals) >= 2 else np.nan
    rows.append({"metric": k, "value": v, "ci_low": lo, "ci_high": hi})

metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

prow = metrics_df[metrics_df["metric"] == primary_metric]
if len(prow):
    pv = float(prow.iloc[0]["value"])
    pci = (float(prow.iloc[0]["ci_low"]), float(prow.iloc[0]["ci_high"]))
    mlca_runtime.record_metric(primary_metric, pv, "test", ci=pci)

if "roc_auc" in point:
    mlca_runtime.record_fact_candidate(
        "test_roc_auc", float(point["roc_auc"]), source="test_metrics.csv"
    )

pred = pd.DataFrame({"row_id": test.index, "partition": "test"})
if len(targets) == 1:
    pred["y_true"] = y_true
    pred["y_pred"] = y_pred
else:
    for i, t in enumerate(targets):
        pred[f"y_true__{t}"] = test[t].to_numpy()
        pred[f"y_pred__{t}"] = y_pred[:, i] if getattr(y_pred, "ndim", 1) > 1 else y_pred
for j, c in enumerate(classes):
    pred[f"proba__{c}"] = proba[:, j]

mlca_runtime.save_row_outputs(pred, "test_predictions.parquet")
print(f"test rows: {len(test)}, features: {len(feat_cols)}")
print(f"{primary_metric}: {pv:.4f} CI [{pci[0]:.4f}, {pci[1]:.4f}]")
