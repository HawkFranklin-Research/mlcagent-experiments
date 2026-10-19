import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, roc_curve,
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
avail = sorted(proto)
for k in ("task", "targets", "excluded_columns"):
    if k not in proto:
        raise ValueError(f"Missing protocol key {k}; available: {avail}")
task = proto["task"]
if "primary_metric" not in task:
    raise ValueError(f"Missing task.primary_metric; task keys: {sorted(task)}")
primary = task["primary_metric"]
targets = proto["targets"]
excluded = set(proto["excluded_columns"])
group = proto.get("group_column")
drop = set(targets) | excluded | ({group} if group else set())

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if not pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) == len(s):
        continue
    feature_cols.append(c)


def model_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        cols = list(obj.feature_names_in_)
    elif isinstance(obj, Pipeline):
        cols = None
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                cols = list(step.feature_names_in_)
                break
        if cols is None:
            cols = list(feature_cols)
    else:
        cols = list(feature_cols)
    bad = set(cols) & drop
    if bad:
        raise ValueError(f"Model input includes target/excluded/group columns: {bad}")
    return cols


def classes_of(obj):
    cls = getattr(obj, "classes_", None)
    if cls is None:
        last = obj.named_steps[list(obj.named_steps)[-1]]
        cls = last.classes_
    return np.asarray(cls)


def evaluate(name, obj):
    cols = model_cols(obj)
    miss = [c for c in cols if c not in val.columns]
    if miss:
        raise ValueError(f"Validation missing columns for {name}: {miss}")
    X = val[cols]
    y = val[targets[0]] if len(targets) == 1 else val[targets]
    proba = obj.predict_proba(X)
    classes = classes_of(obj)
    pred = obj.predict(X)

    if task.get("type") == "multiclass" or len(classes) > 2:
        present = [c for c in classes if np.any(y == c)]
        left = [c for c in classes if c not in present]
        if left:
            print(f"{name} classes absent in val: {left}")
        aucs = []
        for c in present:
            j = int(np.where(classes == c)[0][0])
            aucs.append(roc_auc_score((y == c).astype(int), proba[:, j]))
        metrics = {
            "roc_auc_ovr": float(np.mean(aucs)) if aucs else float("nan"),
            "macro_f1": float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
        }
        thresholds = {k: np.nan for k in metrics}
    else:
        pos = task.get("positive_label")
        if pos is None or pos not in classes:
            pos = classes[1]
        j = int(np.where(classes == pos)[0][0])
        p = proba[:, j]
        yb = (y == pos).astype(int)
        fpr, tpr, thr = roc_curve(yb, p)
        youden = float(thr[np.argmax(tpr - fpr)])
        metrics = {
            "roc_auc": float(roc_auc_score(yb, p)),
            "pr_auc": float(average_precision_score(yb, p)),
            "balanced_accuracy": float(balanced_accuracy_score(yb, (p >= 0.5).astype(int))),
            "brier": float(brier_score_loss(yb, p)),
        }
        thresholds = {k: youden for k in metrics}

    if primary not in metrics:
        raise ValueError(f"Primary metric {primary} not computed; metrics={sorted(metrics)}")
    return [(name, k, float(metrics[k]), float(thresholds[k])) for k in metrics]


models = {
    "logistic": inp["07_model_train_logistic_Q9WJ6BHH/model.pkl"],
    "alternative": inp["08_model_train_alternative_3WAJAX78/alternative_model.pkl"],
}
rows = []
for name, path in models.items():
    with open(path, "rb") as f:
        obj = pickle.load(f)
    rows.extend(evaluate(name, obj))

df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
df["value"] = df["value"].map(lambda v: float(f"{v:.4g}") if pd.notna(v) else v)
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics")

sub = df[df.metric == primary]
ascending = primary in {"brier", "brier_score", "log_loss", "rmse", "mae", "mse"}
best = sub.sort_values("value", ascending=ascending).iloc[0]
mlca_runtime.record_metric(primary, float(best.value), "val")
mlca_runtime.record_fact_candidate("best_model", best.model, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best.value), source="validation_metrics.csv")

print(f"train rows={len(train)} val rows={len(val)}")
print(f"features={len(feature_cols)} models={list(models)}")
print(f"best={best.model} {primary}={best.value:.4g}")
print("wrote validation_metrics.csv")
