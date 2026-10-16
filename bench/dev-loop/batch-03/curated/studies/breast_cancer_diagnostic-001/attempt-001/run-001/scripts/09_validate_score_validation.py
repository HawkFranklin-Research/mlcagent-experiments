import mlca_runtime
import json, pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, roc_curve, f1_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
if "targets" not in protocol:
    raise KeyError(f"Missing 'targets'. Available keys: {list(protocol.keys())}")
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)
task = protocol["task"]
primary_metric = task["primary_metric"]
task_type = task.get("type", "binary")
positive_label = protocol.get("positive_label", task.get("positive_label", None))

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if s.dtype == object or str(s.dtype).startswith("string") or str(s.dtype).startswith("category"):
        if s.nunique() == len(train):
            continue
    if pd.api.types.is_integer_dtype(s):
        if s.nunique() == len(train):
            continue
    feature_cols.append(c)

with open(inputs["05_preprocess_fit_preprocessing_QB1BRK7H/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_select_features_6NMQRFNK/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

def build_pipeline(model_obj, preprocessor, selector):
    if isinstance(model_obj, Pipeline):
        return model_obj
    if isinstance(selector, Pipeline):
        return Pipeline([("selector", selector), ("model", model_obj)])
    steps = []
    if preprocessor is not None:
        steps.append(("preprocessor", preprocessor))
    if selector is not None:
        steps.append(("selector", selector))
    steps.append(("model", model_obj))
    return Pipeline(steps)

def get_classes(model):
    if hasattr(model, "classes_"):
        return model.classes_
    if isinstance(model, Pipeline):
        last = model.steps[-1][1]
        if hasattr(last, "classes_"):
            return last.classes_
    return None

def get_expected_columns(model):
    if isinstance(model, Pipeline):
        first = model.steps[0][1]
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None

def evaluate(pipe, X_val, y_val, task_type, positive_label):
    classes = get_classes(pipe)
    if classes is None:
        classes = np.unique(y_val)
    if task_type == "binary":
        if positive_label is None:
            pos_label = classes[1]
        else:
            pos_label = positive_label
        pos_idx = list(classes).index(pos_label)
        proba = pipe.predict_proba(X_val)
        prob_pos = proba[:, pos_idx]
        y_bin = (y_val == pos_label).astype(int)
        auc = roc_auc_score(y_bin, prob_pos)
        pr_auc = average_precision_score(y_bin, prob_pos)
        brier = brier_score_loss(y_bin, prob_pos)
        fpr, tpr, thresholds = roc_curve(y_bin, prob_pos)
        j = tpr - fpr
        best_idx = np.argmax(j)
        best_thr = thresholds[best_idx]
        if np.isinf(best_thr):
            best_thr = np.max(prob_pos)
        y_pred = (prob_pos >= best_thr).astype(int)
        bal_acc = balanced_accuracy_score(y_bin, y_pred)
        metrics = {"roc_auc": auc, "pr_auc": pr_auc, "balanced_accuracy": bal_acc, "brier_score": brier}
        return metrics, best_thr
    else:
        proba = pipe.predict_proba(X_val)
        y_pred = pipe.predict(X_val)
        auc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)
        f1 = f1_score(y_val, y_pred, average="macro")
        metrics = {"roc_auc_ovr": auc, "f1_macro": f1}
        return metrics, np.nan

models = {
    "logistic": inputs["07_model_train_logistic_1CFM1FE6/model.pkl"],
    "alternative": inputs["08_model_train_alternative_KMR53KBH/alternative_model.pkl"],
}

rows = []
best_model = None
best_val = None
best_auc = None
lower_better = {"brier_score", "log_loss", "mse", "mae"}

for name, path in models.items():
    with open(path, "rb") as f:
        model_obj = pickle.load(f)
    pipe = build_pipeline(model_obj, preprocessor, selector)
    expected = get_expected_columns(pipe)
    if expected is not None:
        expected = [c for c in expected if c in feature_cols]
        X_val = val[expected] if expected else val[feature_cols]
    else:
        X_val = val[feature_cols]
    y_val = val[targets[0]]
    metrics, thr = evaluate(pipe, X_val, y_val, task_type, positive_label)
    for metric, value in metrics.items():
        rows.append({"model": name, "metric": metric, "value": value, "threshold": thr})
    mlca_runtime.record_metric(primary_metric, metrics[primary_metric], "val")
    val_metric = metrics[primary_metric]
    if best_val is None:
        best_val = val_metric
        best_model = name
        best_auc = metrics.get("roc_auc", metrics.get("roc_auc_ovr"))
    else:
        if primary_metric in lower_better:
            if val_metric < best_val:
                best_val = val_metric
                best_model = name
                best_auc = metrics.get("roc_auc", metrics.get("roc_auc_ovr"))
        else:
            if val_metric > best_val:
                best_val = val_metric
                best_model = name
                best_auc = metrics.get("roc_auc", metrics.get("roc_auc_ovr"))

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics for candidate models")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")
print(f"train shape: {train.shape}, val shape: {val.shape}")
print(f"best model: {best_model}, best val roc_auc: {best_auc:.4f}")
