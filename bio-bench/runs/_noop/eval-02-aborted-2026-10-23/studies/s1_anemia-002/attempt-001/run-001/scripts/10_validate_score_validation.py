import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, roc_curve
)
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
task = protocol["task"]
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")
primary = task["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
target_col = targets[0]

def is_keylike(s):
    name = str(s.name).lower()
    vals = s.notna()
    if vals.sum() <= 1:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == vals.sum()
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == vals.sum() and (
            s.is_monotonic_increasing or any(k in name for k in ["id", "key", "code", "index"])
        )
    return False

skip = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [
    c for c in train.columns
    if c not in skip and not is_keylike(train[c])
]

def load_pkl(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("06_preprocess_fit_preprocessing_FTEQSYVR/preprocessor.pkl")
sel = load_pkl("07_features_retain_features_C5JWX6VS/selector_all.pkl")
candidates = [
    ("logistic", load_pkl("08_model_train_logistic_6DPRW9XX/model.pkl")),
    ("alternative", load_pkl("09_model_train_alternative_4BGY008E/alternative_model.pkl")),
]

def classes_of(obj):
    if hasattr(obj, "classes_"):
        return np.asarray(obj.classes_)
    if isinstance(obj, Pipeline):
        for step in reversed(list(obj.named_steps.values())):
            if hasattr(step, "classes_"):
                return np.asarray(step.classes_)
    raise AttributeError("model classes_ not found")

def input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline) and obj.named_steps:
        first = next(iter(obj.named_steps.values()))
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
    return feature_cols

def frontend_cols(front):
    if hasattr(front, "feature_names_in_"):
        return list(front.feature_names_in_)
    if isinstance(front, Pipeline) and front.named_steps:
        first = next(iter(front.named_steps.values()))
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
    return feature_cols

if isinstance(sel, Pipeline):
    frontend = sel
else:
    sel_cols = getattr(sel, "feature_names_in_", None)
    if sel_cols is not None and set(sel_cols).issubset(set(feature_cols)):
        frontend = sel
    else:
        frontend = Pipeline([("preprocessor", pre), ("selector", sel)])

def predict_proba(obj, frame):
    cols = [c for c in input_cols(obj) if c in frame.columns]
    try:
        return obj.predict_proba(frame[cols])
    except Exception:
        if isinstance(obj, Pipeline):
            raise
        fcols = [c for c in frontend_cols(frontend) if c in frame.columns]
        transformed = frontend.transform(frame[fcols])
        return obj.predict_proba(transformed)

rows = []
primary_metrics = []
best_name, best_auc = None, -np.inf
y_val = val[target_col].to_numpy()

for name, model in candidates:
    proba = predict_proba(model, val)
    classes = classes_of(model)
    pos_label = task.get("positive_label", protocol.get("positive_label"))
    if pos_label is None:
        pos_label = classes[1]
    pos_idx = int(np.where(classes == pos_label)[0][0])
    y_true = (y_val == pos_label).astype(int)
    y_score = proba[:, pos_idx]

    auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_thr = float(thresholds[np.argmax(tpr - fpr)])
    y_pred = (y_score >= youden_thr).astype(int)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_score))

    rows.extend([
        (name, primary, auc, np.nan),
        (name, "pr_auc", pr_auc, np.nan),
        (name, "balanced_accuracy", bal_acc, youden_thr),
        (name, "brier_score", brier, np.nan),
    ])
    primary_metrics.append((name, auc))
    if auc > best_auc:
        best_name, best_auc = name, auc

def sig4(x):
    return float(f"{x:.4g}") if pd.notna(x) else np.nan

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
metrics_df["value"] = metrics_df["value"].map(sig4)
metrics_df["threshold"] = metrics_df["threshold"].map(sig4)
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics by candidate model")

for _, auc in primary_metrics:
    mlca_runtime.record_metric(primary, auc, "val")
mlca_runtime.record_metric(primary, best_auc, "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")

print(f"train={train.shape} val={val.shape}")
print(f"best_model={best_name} {primary}={best_auc:.4f}")
