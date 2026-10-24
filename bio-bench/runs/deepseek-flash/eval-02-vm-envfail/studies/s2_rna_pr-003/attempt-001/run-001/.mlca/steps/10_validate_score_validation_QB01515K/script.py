import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, roc_curve, f1_score,
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"missing protocol key {key}; available: {list(protocol)}")

task = protocol["task"]
primary_metric = task["primary_metric"]
targets = list(protocol["targets"])
excluded = protocol["excluded_columns"]
if isinstance(excluded, dict):
    excluded = excluded.get("columns") or excluded.get("entries") or excluded.get("examples", [])
excluded = set(excluded)
group_col = protocol.get("group_column")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
y_col = targets[0]
y_val = val[y_col]

feature_cols = [
    c for c in train.columns
    if c not in set(targets) and c not in excluded and c != group_col
]
key_like = []
for c in feature_cols:
    s = train[c]
    if s.nunique(dropna=False) != len(s):
        continue
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        key_like.append(c)
    elif pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        name = str(c).lower()
        if s.is_monotonic_increasing or s.is_monotonic_decreasing or name.endswith("id") or name == "id":
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in set(key_like)]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inp["06_preprocess_fit_preprocessing_1X70GNWA/preprocessor.pkl"])
sel = load_pkl(inp["07_features_select_features_41RKQQBB/selector.pkl"])
raw_models = {
    "logistic": load_pkl(inp["08_model_train_logistic_Z47S9AWS/model.pkl"]),
    "alternative": load_pkl(inp["09_model_train_alternative_JTCWES9Z/alternative_model.pkl"]),
}

def get_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        try:
            cols = list(obj.feature_names_in_)
            if cols:
                return cols
        except Exception:
            pass
    if isinstance(obj, Pipeline):
        steps = list(obj.named_steps.values())
        if steps and hasattr(steps[0], "feature_names_in_"):
            return list(steps[0].feature_names_in_)
    return None

def prepare(obj):
    if isinstance(obj, Pipeline) and get_cols(obj) is not None:
        return obj
    if hasattr(obj, "predict_proba") and get_cols(obj) is not None:
        return obj
    if hasattr(obj, "predict_proba"):
        sel_cols = get_cols(sel) if sel is not None else None
        if sel is not None and sel_cols is not None and set(sel_cols) == set(feature_cols):
            return Pipeline([("selector", sel), ("model", obj)])
        if pre is not None and sel is not None:
            return Pipeline([("preprocessor", pre), ("selector", sel), ("model", obj)])
        if pre is not None:
            return Pipeline([("preprocessor", pre), ("model", obj)])
    return obj

def get_classes(obj):
    if hasattr(obj, "classes_"):
        return np.asarray(obj.classes_)
    if isinstance(obj, Pipeline):
        last = list(obj.named_steps.values())[-1]
        if hasattr(last, "classes_"):
            return np.asarray(last.classes_)
    return None

def metric_value(metric, vals):
    if metric in vals:
        return vals[metric]
    if metric in ("roc_auc", "roc_auc_ovr", "auc", "roc_auc_score"):
        return vals.get("roc_auc")
    if metric in ("f1", "f1_macro", "f1_score"):
        return vals.get("f1_macro")
    return np.nan

prepared_models = {name: prepare(obj) for name, obj in raw_models.items()}
first_classes = get_classes(next(iter(prepared_models.values())))
is_multiclass = str(task.get("type", "binary")).lower().startswith("multi") or int(task.get("n_classes", 2)) > 2

positive_label = None
if not is_multiclass:
    positive_label = task.get("positive_label") or protocol.get("positive_label")
    if positive_label is None and first_classes is not None:
        positive_label = first_classes[1]
    if positive_label is None:
        raise ValueError("cannot determine positive label")

rows = []
primary_by_model = {}
for name, model in prepared_models.items():
    cols = get_cols(model) or feature_cols
    missing = [c for c in cols if c not in val.columns]
    if missing:
        raise KeyError(f"validation missing model columns: {missing[:5]}")
    X_val = val.loc[:, cols]
    proba = model.predict_proba(X_val)
    classes = get_classes(model)

    if not is_multiclass:
        if classes is None:
            classes = np.asarray([0, 1])
        if positive_label not in classes:
            raise ValueError(f"positive label {positive_label!r} not in model classes {classes!r}")
        pos_idx = int(np.where(classes == positive_label)[0][0])
        proba_pos = proba[:, pos_idx]
        y_bin = (y_val == positive_label).astype(int).to_numpy()
        fpr, tpr, thresholds = roc_curve(y_bin, proba_pos)
        thr = float(thresholds[int(np.argmax(tpr - fpr))])
        pred_bin = (proba_pos >= thr).astype(int)
        vals = {
            "roc_auc": roc_auc_score(y_bin, proba_pos),
            "pr_auc": average_precision_score(y_bin, proba_pos),
            "balanced_accuracy": balanced_accuracy_score(y_bin, pred_bin),
            "brier_score": brier_score_loss(y_bin, proba_pos),
        }
        metric_order = []
        for m in (primary_metric, "roc_auc", "pr_auc", "balanced_accuracy", "brier_score"):
            if m and m not in metric_order:
                v = metric_value(m, vals)
                if pd.notna(v):
                    metric_order.append(m)
        for m in metric_order:
            rows.append({"model": name, "metric": m, "value": metric_value(m, vals), "threshold": thr})
        primary_by_model[name] = metric_value(primary_metric, vals)
    else:
        present = [c for c in classes if np.any(y_val.to_numpy() == c)] if classes is not None else []
        if classes is not None and len(present) == len(classes):
            roc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)
        else:
            aucs = []
            for c in present:
                j = int(np.where(classes == c)[0][0])
                aucs.append(roc_auc_score((y_val == c).astype(int), proba[:, j]))
            roc = float(np.mean(aucs)) if aucs else np.nan
            if classes is not None:
                print("classes left out", [c for c in classes if c not in present])
        pred = model.predict(X_val)
        vals = {"roc_auc": roc, "f1_macro": f1_score(y_val, pred, average="macro")}
        metric_order = []
        for m in (primary_metric, "roc_auc", "f1_macro"):
            if m and m not in metric_order:
                v = metric_value(m, vals)
                if pd.notna(v):
                    metric_order.append(m)
        for m in metric_order:
            rows.append({"model": name, "metric": m, "value": metric_value(m, vals), "threshold": np.nan})
        primary_by_model[name] = metric_value(primary_metric, vals)

df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
df["value"] = df["value"].map(lambda x: np.nan if pd.isna(x) else float(f"{x:.4g}"))
df["threshold"] = df["threshold"].map(lambda x: np.nan if pd.isna(x) else float(f"{x:.4g}"))
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics by model")

higher_better = primary_metric not in ("brier_score", "log_loss", "mae", "mse", "rmse")
best_name = max(primary_by_model, key=primary_by_model.get) if higher_better else min(primary_by_model, key=primary_by_model.get)
best_val = float(primary_by_model[best_name])
mlca_runtime.record_metric(primary_metric, best_val, split="val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")

best_roc = df[(df["model"] == best_name) & (df["metric"] == "roc_auc")]
if len(best_roc):
    mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best_roc["value"].iloc[0]), source="validation_metrics.csv")

print(f"features={len(feature_cols)} models={len(prepared_models)}")
print(f"best_model={best_name} {primary_metric}={best_val:.4g}")
