import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, roc_curve, f1_score
)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
task = proto["task"]
primary = task["primary_metric"]
targets = proto["targets"]
excluded = set(proto.get("excluded_columns", []))
group = proto.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
target = targets[0]

def is_keylike(s):
    if s.dtype == object or str(s.dtype).startswith("string") or str(s.dtype) == "category":
        return s.nunique(dropna=True) == len(s) and s.notna().all()
    return False

excl = set(targets) | excluded
if group:
    excl.add(group)
features = [c for c in train.columns if c not in excl and not is_keylike(train[c])]
missing = [c for c in features if c not in val.columns]
if missing:
    raise KeyError(f"Missing feature columns in val: {missing}")

pre = pickle.load(open(inputs["05_preprocess_fit_preprocessing_7578M7XP/preprocessor.pkl"], "rb"))
sel = pickle.load(open(inputs["06_features_select_features_J83AE8FG/selector.pkl"], "rb"))
models = {
    "logistic": pickle.load(open(inputs["07_model_train_logistic_X39FZFWK/model.pkl"], "rb")),
    "alternative": pickle.load(open(inputs["08_model_train_alternative_XM6RHY82/alternative_model.pkl"], "rb")),
}

def first_input_cols(obj):
    cols = getattr(obj, "feature_names_in_", None)
    if cols is not None:
        return list(cols)
    if isinstance(obj, Pipeline):
        first = obj.steps[0][1]
        cols = getattr(first, "feature_names_in_", None)
        if cols is not None:
            return list(cols)
    return None

def accepts_raw(obj):
    cols = first_input_cols(obj)
    return cols is not None and set(cols) == set(features)

def final_estimator(pipe):
    obj = pipe
    while isinstance(obj, Pipeline):
        obj = obj.steps[-1][1]
    return obj

def make_predictor(model):
    if accepts_raw(model) or (isinstance(model, Pipeline) and first_input_cols(model) is None):
        return model, first_input_cols(model) or features
    if accepts_raw(sel):
        front, cols = sel, first_input_cols(sel) or features
    else:
        front = Pipeline([("preprocessor", pre), ("selector", sel)])
        cols = first_input_cols(pre) or features
    return Pipeline([("frontend", front), ("model", model)]), cols

rows = []
model_primary = {}
for name, model in models.items():
    pipe, cols = make_predictor(model)
    X = val[cols]
    proba = pipe.predict_proba(X)
    classes = final_estimator(pipe).classes_
    y = val[target].to_numpy()
    if task.get("type") == "binary" or len(classes) == 2:
        pos_label = task.get("positive_label", classes[1])
        pos_idx = int(np.where(classes == pos_label)[0][0])
        y_bin = (y == pos_label).astype(int)
        p = proba[:, pos_idx]
        auc = float(roc_auc_score(y_bin, p))
        pr_auc = float(average_precision_score(y_bin, p))
        brier = float(brier_score_loss(y_bin, p))
        fpr, tpr, thr = roc_curve(y_bin, p)
        best_i = int(np.argmax(tpr - fpr))
        threshold = float(thr[best_i])
        if not np.isfinite(threshold):
            threshold = float(np.max(p))
        y_pred = (p >= threshold).astype(int)
        bal = float(balanced_accuracy_score(y_bin, y_pred))
        metrics = [("roc_auc", auc), ("pr_auc", pr_auc), ("balanced_accuracy", bal), ("brier", brier)]
        model_primary[name] = auc if primary == "roc_auc" else dict(metrics).get(primary, np.nan)
        for metric, value in metrics:
            rows.append({"model": name, "metric": metric, "value": value, "threshold": threshold})
    else:
        present = [c for c in classes if c in set(y)]
        aucs = []
        for j, c in enumerate(classes):
            if c in present:
                aucs.append(roc_auc_score((y == c).astype(int), proba[:, j]))
        auc = float(np.mean(aucs)) if aucs else np.nan
        pred = classes[np.argmax(proba, axis=1)]
        f1 = float(f1_score(y, pred, average="macro"))
        model_primary[name] = auc if primary == "roc_auc" else np.nan
        rows.extend([
            {"model": name, "metric": "roc_auc_ovr_macro", "value": auc, "threshold": np.nan},
            {"model": name, "metric": "f1_macro", "value": f1, "threshold": np.nan},
        ])

metrics_df = pd.DataFrame(rows)[["model", "metric", "value", "threshold"]]
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation candidate metrics")

lower_better = {"brier", "log_loss", "mse", "rmse", "mae"}
direction = -1 if primary.lower() in lower_better else 1
best_model = max(model_primary, key=lambda k: direction * model_primary[k])
best_val = float(model_primary[best_model])
mlca_runtime.record_metric(primary, best_val, split="val")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
auc_row = metrics_df[(metrics_df["model"] == best_model) & (metrics_df["metric"] == "roc_auc")]
if not auc_row.empty:
    mlca_runtime.record_fact_candidate("best_val_roc_auc", float(auc_row.iloc[0]["value"]), source="validation_metrics.csv")

print(f"val_rows={len(val)} models={len(models)} features={len(features)}")
print(f"best_model={best_model} {primary}={best_val:.4f}")
