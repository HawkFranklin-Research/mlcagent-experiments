import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, roc_curve

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

required = ["targets", "excluded_columns", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {sorted(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
task = protocol["task"]
primary_metric = task["primary_metric"]
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

exclude_set = set(targets) | set(excluded) | ({group} if group else set())

def is_key_like(s):
    if pd.api.types.is_float_dtype(s):
        return False
    if s.nunique(dropna=False) != len(s):
        return False
    return pd.api.types.is_integer_dtype(s) or pd.api.types.is_string_dtype(s) or s.dtype.kind in "OUS"

feature_cols = [c for c in train.columns if c not in exclude_set]
feature_cols = [c for c in feature_cols if not is_key_like(train[c])]
X_val = val[feature_cols].copy()
y_true = val[target].to_numpy()

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inp["05_preprocess_fit_preprocessing_6KJVP4ST/preprocessor.pkl"])
sel = load_pkl(inp["06_features_select_features_TX9DR60B/selector.pkl"])
model_items = [
    ("logistic", load_pkl(inp["07_model_train_logistic_GD4X7DTC/model.pkl"])),
    ("alternative", load_pkl(inp["08_model_train_alternative_0A2EAXQK/alternative_model.pkl"])),
]

def predict_with(obj, X):
    X2 = X
    if hasattr(obj, "feature_names_in_"):
        names = list(obj.feature_names_in_)
        if all(n in X.columns for n in names):
            X2 = X[names]
    return obj.predict_proba(X2)

def predict_proba_any(model_obj, X, pre, sel):
    candidates = [model_obj]
    if sel is not None:
        candidates.append(Pipeline([("selector", sel), ("model", model_obj)]))
    if pre is not None and sel is not None:
        candidates.append(Pipeline([("preprocessor", pre), ("selector", sel), ("model", model_obj)]))
    if pre is not None:
        candidates.append(Pipeline([("preprocessor", pre), ("model", model_obj)]))
    last_err = None
    for obj in candidates:
        try:
            return obj, predict_with(obj, X)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"No working pipeline: {last_err}")

def classes_of(obj):
    if hasattr(obj, "classes_"):
        return np.asarray(obj.classes_)
    if hasattr(obj, "steps"):
        return classes_of(obj.steps[-1][1])
    return None

positive_label = task.get("positive_label", task.get("positive_class", None))
rows = []
for name, model_obj in model_items:
    fitted, proba = predict_proba_any(model_obj, X_val, pre, sel)
    classes = classes_of(fitted)
    if positive_label is None:
        if classes is None or len(classes) != 2:
            raise ValueError("Cannot infer positive label")
        positive_label = classes[1]
    idx = int(np.where(classes == positive_label)[0][0]) if classes is not None else 1
    p = proba[:, idx]
    y_bin = (y_true == positive_label).astype(int)

    roc = roc_auc_score(y_bin, p)
    pr = average_precision_score(y_bin, p)
    brier = brier_score_loss(y_bin, p)
    fpr, tpr, thresholds = roc_curve(y_bin, p)
    thr = float(thresholds[int(np.argmax(tpr - fpr))])
    bal = balanced_accuracy_score(y_bin, (p >= thr).astype(int))

    rows.extend([
        {"model": name, "metric": primary_metric, "value": roc, "threshold": np.nan},
        {"model": name, "metric": "pr_auc", "value": pr, "threshold": np.nan},
        {"model": name, "metric": "balanced_accuracy", "value": bal, "threshold": thr},
        {"model": name, "metric": "brier", "value": brier, "threshold": np.nan},
    ])

df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])

def sig4(x):
    if pd.isna(x):
        return np.nan
    return float(f"{float(x):.4g}")

df["value"] = df["value"].map(sig4)
df["threshold"] = df["threshold"].map(sig4)
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics for candidate models.")

higher_is_better = primary_metric in {
    "roc_auc", "pr_auc", "balanced_accuracy", "f1", "f1_macro", "accuracy", "r2", "roc_auc_ovr"
}
prim = df[df["metric"] == primary_metric].sort_values("value", ascending=not higher_is_better)
best_row = prim.iloc[0]
best_model = best_row["model"]
best_value = float(best_row["value"])
best_auc = float(df[(df["model"] == best_model) & (df["metric"] == primary_metric)]["value"].iloc[0])

mlca_runtime.record_metric(primary_metric, best_value, "val")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")

print(f"features={len(feature_cols)} train={train.shape} val={val.shape}")
print(f"best_model={best_model} {primary_metric}={best_value:.4g}")
