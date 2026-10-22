import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, recall_score
)
import mlca_runtime

SEED = 42


def req(d, k):
    if k not in d:
        raise KeyError(f"Missing required key '{k}'; available top-level keys: {list(d.keys())}")
    return d[k]


def search_threshold(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if "threshold" in kl:
                if isinstance(v, (int, float, np.number)):
                    return float(v)
                if isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, (int, float, np.number)):
                            return float(vv)
            r = search_threshold(v)
            if r is not None:
                return r
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            r = search_threshold(v)
            if r is not None:
                return r
    return None


def threshold_from_df(df):
    for c in df.columns:
        if "threshold" in str(c).lower():
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals):
                return float(vals.iloc[0])
    if "metric" in df.columns:
        m = df["metric"].astype(str).str.lower().str.contains("threshold")
        if m.any():
            row = df[m].iloc[0]
            for c in ["value", "threshold", "val"]:
                if c in row.index and pd.notna(row[c]):
                    try:
                        return float(row[c])
                    except Exception:
                        pass
    return None


def threshold_from_model(model):
    objs = [model]
    if hasattr(model, "named_steps"):
        objs += list(model.named_steps.values())
    for obj in objs:
        for k, v in vars(obj).items():
            if "threshold" in k.lower() and isinstance(v, (int, float, np.number)):
                return float(v)
    return None


def metrics_on_sample(yb, score, ypred):
    out = {}
    if len(np.unique(yb)) > 1:
        out["roc_auc"] = roc_auc_score(yb, score)
        out["pr_auc"] = average_precision_score(yb, score)
    out["balanced_accuracy"] = balanced_accuracy_score(yb, ypred)
    out["brier_score"] = brier_score_loss(yb, score)
    if yb.sum() > 0:
        out["sensitivity"] = recall_score(yb, ypred, pos_label=1, zero_division=0)
    if (1 - yb).sum() > 0:
        out["specificity"] = recall_score(yb, ypred, pos_label=0, zero_division=0)
    return out


def sig(x, n=4):
    if pd.isna(x):
        return np.nan
    return float(f"{float(x):.{n}g}")


mlca_runtime.set_seed(SEED)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
task = req(protocol, "task")
primary_metric = req(task, "primary_metric")
target = targets[0]

with inp["latest_model"].open("rb") as f:
    model = pickle.load(f)
if isinstance(model, dict) and "model" in model:
    model = model["model"]

feat_cols = None
if hasattr(model, "feature_names_in_"):
    feat_cols = list(model.feature_names_in_)
else:
    train = pd.read_parquet(inp["train"])
    group = protocol.get("group_column")
    drop = set(targets) | set(excluded)
    if group in train.columns:
        drop.add(group)
    feat_cols = [c for c in train.columns if c not in drop]
    feat_cols = [c for c in feat_cols if not (
        train[c].dtype == object and train[c].nunique(dropna=True) == len(train) > 1
    )]

forbidden = set(targets) | set(excluded)
if forbidden.intersection(feat_cols):
    raise ValueError("Feature columns include target or excluded columns.")

test = pd.read_parquet(inp["test"])
X_test = test[list(feat_cols)]
y_true = test[target]

pos_label = task.get("positive_label", protocol.get("positive_label"))
if pos_label is None:
    pos_label = model.classes_[1]
classes = list(model.classes_)
pos_idx = classes.index(pos_label)
proba = model.predict_proba(X_test)
score = proba[:, pos_idx]

try:
    vdf = pd.read_parquet(inp["09_validate_score_validation_M840FB6A/validation_metrics.parquet"])
except Exception:
    vdf = pd.read_csv(inp["09_validate_score_validation_M840FB6A/validation_metrics.csv"])
threshold = threshold_from_df(vdf)
if threshold is None:
    vman = json.loads(inp["09_validate_score_validation_M840FB6A/manifest.json"].read_text())
    threshold = search_threshold(vman)
if threshold is None:
    threshold = threshold_from_model(model)
if threshold is None:
    threshold = 0.5

yb = (y_true == pos_label).astype(int).to_numpy()
ypred = (score >= threshold).astype(int)
point = metrics_on_sample(yb, score, ypred)

metric_names = [primary_metric] + [
    m for m in ["roc_auc", "pr_auc", "balanced_accuracy", "brier_score", "sensitivity", "specificity"]
    if m != primary_metric
]
rng = np.random.default_rng(SEED)
boot = {m: [] for m in metric_names}
n = len(yb)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    bm = metrics_on_sample(yb[idx], score[idx], ypred[idx])
    for m in metric_names:
        if m in bm:
            boot[m].append(bm[m])

rows = []
for m in metric_names:
    vals = np.asarray(boot[m], dtype=float)
    lo = np.nanpercentile(vals, 2.5) if len(vals) else np.nan
    hi = np.nanpercentile(vals, 97.5) if len(vals) else np.nan
    rows.append({
        "metric": m,
        "value": sig(point.get(m, np.nan)),
        "ci_low": sig(lo),
        "ci_high": sig(hi),
    })

metrics_df = pd.DataFrame(rows, columns=["metric", "value", "ci_low", "ci_high"])
mlca_runtime.save_table(metrics_df, "test_metrics.csv")

primary_row = metrics_df.loc[metrics_df["metric"] == primary_metric].iloc[0]
mlca_runtime.record_metric(
    primary_metric, float(primary_row["value"]), "test",
    ci=(float(primary_row["ci_low"]), float(primary_row["ci_high"]))
)
if primary_metric == "roc_auc":
    mlca_runtime.record_fact_candidate(
        "test_roc_auc", float(primary_row["value"]), source="test_metrics.csv"
    )

pred = pd.DataFrame({
    "row_id": test.index,
    "partition": "test",
    "y_true": y_true.to_numpy(),
    "y_pred": np.where(ypred == 1, pos_label, classes[1 - pos_idx]),
})
for j, c in enumerate(classes):
    pred[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred, "test_predictions.parquet")

print(f"test rows={len(test)}, features={len(feat_cols)}, threshold={threshold:.4g}")
print(f"{primary_metric}={primary_row['value']:.4g} CI=({primary_row['ci_low']:.4g}, {primary_row['ci_high']:.4g})")
