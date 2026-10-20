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

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing {key}; available top-level keys: {sorted(protocol.keys())}")

targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
task = protocol["task"]
primary = task["primary_metric"]
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
exclude_set = set(targets) | excluded | ({group} if group else set())

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
models = [
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

def predict_any(model_obj, X, pre, sel):
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

declared_pos = task.get("positive_label", task.get("positive_class", None))
higher = {"roc_auc", "pr_auc", "balanced_accuracy", "f1", "f1_macro", "accuracy", "r2", "roc_auc_ovr", "average_precision"}
lower = {"brier", "log_loss", "mae", "mse", "rmse"}
if primary not in higher and primary not in lower:
    raise ValueError(f"Unknown primary metric direction: {primary}")

rows = []
for name, model_obj in models:
    fitted, proba = predict_any(model_obj, X_val, pre, sel)
    classes = classes_of(fitted)
    if classes is None or len(classes) != 2:
        raise ValueError(f"Expected binary model classes for {name}, got {classes}")
    pos_label = declared_pos if declared_pos is not None else classes[1]
    idx_arr = np.where(classes == pos_label)[0]
    if len(idx_arr) == 0:
        raise ValueError(f"positive label {pos_label!r} not in {name} classes {classes.tolist()}")
    p = proba[:, int(idx_arr[0])]
    y_bin = (y_true == pos_label).astype(int)

    roc = float(roc_auc_score(y_bin, p))
    pr = float(average_precision_score(y_bin, p))
    brier = float(brier_score_loss(y_bin, p))
    fpr, tpr, thr = roc_curve(y_bin, p)
    finite = np.isfinite(thr)
    if finite.any():
        j = int(np.argmax((tpr - fpr)[finite]))
        youden_thr = float(thr[finite][j])
    else:
        youden_thr = float(thr[int(np.argmax(tpr - fpr))])
    bal = float(balanced_accuracy_score(y_bin, (p >= youden_thr).astype(int)))

    vals = {"roc_auc": roc, "pr_auc": pr, "balanced_accuracy": bal, "brier": brier}
    prim_val = vals.get(primary, roc)
    mlca_runtime.record_metric(primary, prim_val, "val")

    rows.extend([
        {"model": name, "metric": primary, "value": prim_val, "threshold": np.nan},
        {"model": name, "metric": "pr_auc", "value": pr, "threshold": np.nan},
        {"model": name, "metric": "balanced_accuracy", "value": bal, "threshold": youden_thr},
        {"model": name, "metric": "brier", "value": brier, "threshold": np.nan},
    ])

df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
def sig4(x):
    return np.nan if pd.isna(x) else float(f"{float(x):.4g}")
df["value"] = df["value"].map(sig4)
df["threshold"] = df["threshold"].map(sig4)
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics for candidate models.")

prim_df = df[df["metric"] == primary]
best_row = prim_df.sort_values("value", ascending=(primary in lower)).iloc[0]
best_model = best_row["model"]
best_auc = float(df[(df["model"] == best_model) & (df["metric"] == "roc_auc")]["value"].iloc[0])
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")
print(f"features={len(feature_cols)} train={train.shape} val={val.shape}")
print(f"best_model={best_model} {primary}={float(best_row['value']):.4g}")
