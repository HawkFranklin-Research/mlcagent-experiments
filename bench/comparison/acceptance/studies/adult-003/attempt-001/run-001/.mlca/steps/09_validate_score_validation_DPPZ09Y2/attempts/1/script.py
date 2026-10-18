import mlca_runtime, json, pickle, numpy as np, pandas as pd
from math import floor, log10
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, roc_curve, f1_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
task = protocol["task"]
primary_metric = task["primary_metric"]
group_col = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_key_like(s):
    if len(s) < len(train) or s.nunique(dropna=True) != len(train):
        return False
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)

blocked = set(targets) | set(excluded) | ({group_col} if group_col else set())
feature_cols = [c for c in train.columns if c not in blocked]
feature_cols = [c for c in feature_cols if not is_key_like(train[c])]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def ensure_pipeline(obj):
    if isinstance(obj, Pipeline):
        return obj
    pre = load_pkl(inputs["05_preprocess_fit_preprocessing_599QM3CC/preprocessor.pkl"])
    sel = load_pkl(inputs["06_features_select_features_Y69A7R1X/selector.pkl"])
    if isinstance(sel, Pipeline):
        return Pipeline(list(sel.steps) + [("model", obj)])
    return Pipeline([("preprocessor", pre), ("selector", sel), ("model", obj)])

models = {
    "logistic": ensure_pipeline(load_pkl(inputs["07_model_train_logistic_8BYRNT7K/model.pkl"])),
    "alternative": ensure_pipeline(load_pkl(inputs["08_model_train_alternative_5MJCNH4M/alternative_model.pkl"])),
}

def get_classes(model):
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    last = list(model.named_steps)[-1]
    return np.asarray(model.named_steps[last].classes_)

def expected_columns(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if isinstance(model, Pipeline):
        for _, step in model.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return feature_cols

def round_sig(x, sig=4):
    if x is None or (isinstance(x, float) and np.isnan(x)) or x == 0:
        return x
    return round(x, -int(floor(log10(abs(x)))) + (sig - 1))

alias = {
    "roc_auc": "roc_auc", "roc_auc_score": "roc_auc", "auc": "roc_auc",
    "pr_auc": "pr_auc", "average_precision": "pr_auc",
    "balanced_accuracy": "balanced_accuracy", "brier": "brier",
    "brier_score_loss": "brier", "f1_macro": "f1_macro", "roc_auc_ovr": "roc_auc_ovr",
}
primary_key = alias.get(primary_metric, primary_metric)
lower_better = primary_metric in {"brier", "brier_score_loss", "log_loss", "mse", "mae", "rmse"}

rows, model_primary, model_auc = [], {}, {}
for name, model in models.items():
    cols = expected_columns(model)
    bad = set(cols) & blocked
    if bad:
        raise ValueError(f"Model {name} expects target/excluded/group columns: {sorted(bad)}")
    missing = [c for c in cols if c not in val.columns]
    if missing:
        raise KeyError(f"Missing validation columns for {name}: {missing}")
    X_val = val.loc[:, cols].copy()
    y_val = val[target]
    classes = get_classes(model)
    proba = model.predict_proba(X_val)

    if len(classes) == 2:
        pos_label = task.get("positive_label", protocol.get("positive_label", None))
        if pos_label is None or pos_label not in classes:
            pos_label = classes[1]
        pos_idx = int(np.where(classes == pos_label)[0][0])
        y_bin = (y_val == pos_label).astype(int).to_numpy()
        p = proba[:, pos_idx]
        roc = roc_auc_score(y_bin, p)
        pr = average_precision_score(y_bin, p)
        fpr, tpr, thr = roc_curve(y_bin, p)
        j = tpr - fpr
        finite = np.where(np.isfinite(thr))[0]
        best_idx = finite[np.argmax(j[finite])] if len(finite) else int(np.argmax(j))
        threshold = float(thr[best_idx])
        bal = balanced_accuracy_score(y_bin, (p >= threshold).astype(int))
        brier = brier_score_loss(y_bin, p)
        metrics = {"roc_auc": roc, "pr_auc": pr, "balanced_accuracy": bal, "brier": brier}
        threshold_value = threshold
    else:
        roc = roc_auc_score(y_val, proba, labels=classes, multi_class="ovr", average="macro")
        f1 = f1_score(y_val, model.predict(X_val), average="macro")
        metrics = {"roc_auc_ovr": roc, "f1_macro": f1}
        threshold_value = np.nan

    pv = metrics.get(primary_key)
    if pv is None:
        raise KeyError(f"Primary metric {primary_metric!r} cannot be computed; available metrics: {list(metrics)}")
    model_primary[name] = float(pv)
    model_auc[name] = float(metrics.get("roc_auc", metrics.get("roc_auc_ovr", np.nan)))

    rows.append({
        "model": name, "metric": primary_metric, "value": float(pv),
        "threshold": threshold_value if primary_metric == "balanced_accuracy" and len(classes) == 2 else np.nan,
    })
    for m, v in metrics.items():
        if m == primary_key or m == primary_metric:
            continue
        rows.append({
            "model": name, "metric": m, "value": float(v),
            "threshold": threshold_value if m == "balanced_accuracy" and len(classes) == 2 else np.nan,
        })

results = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
results["value"] = results["value"].map(round_sig)
results["threshold"] = results["threshold"].map(round_sig)
mlca_runtime.save_table(results, "validation_metrics.csv", caption="Validation metrics for candidate models")

best_name = min(model_primary, key=model_primary.get) if lower_better else max(model_primary, key=model_primary.get)
for name, value in model_primary.items():
    mlca_runtime.record_metric(primary_metric, float(value), split="val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(model_auc[best_name]), source="validation_metrics.csv")

print(f"train rows={len(train)}, val rows={len(val)}")
print(f"models={list(models)}, primary_metric={primary_metric}")
print(f"best_model={best_name}, best_val_{primary_metric}={model_primary[best_name]:.6g}")
