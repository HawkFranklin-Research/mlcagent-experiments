import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

for k in ["targets", "excluded_columns", "task"]:
    if k not in protocol:
        raise ValueError(f"Missing protocol key {k!r}; available top-level keys: {list(protocol.keys())}")

targets = protocol["targets"] if isinstance(protocol["targets"], list) else [protocol["targets"]]
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")
task = protocol["task"]
primary = task["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
if len(targets) != 1 or targets[0] not in train.columns or targets[0] not in val.columns:
    raise ValueError("Expected exactly one declared target present in train and val")

def is_key_like(col):
    s = train[col]
    if s.isna().any() or s.nunique(dropna=False) != len(s):
        return False
    return bool(
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_integer_dtype(s)
    )

exclude = set(targets) | set(excluded)
if group_col:
    exclude.add(group_col)
raw_features = [c for c in train.columns if c not in exclude and not is_key_like(c)]
if not raw_features:
    raise ValueError("No feature columns remain after exclusions")

def load_pkl(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        for k in ["pipeline", "model", "estimator", "preprocessor", "selector"]:
            if k in obj:
                return obj[k]
    return obj

pre = load_pkl(inp["06_preprocess_fit_preprocessing_1DA5F11K/preprocessor.pkl"])
sel = load_pkl(inp["07_features_select_features_W8F8XFJ6/selector.pkl"])
m08 = load_pkl(inp["08_model_train_logistic_5N3CJC0T/model.pkl"])
m09 = load_pkl(inp["09_model_train_alternative_RPF0Y33W/alternative_model.pkl"])
candidates = [("logistic_08", m08), ("alternative_09", m09)]

def final_estimator(obj):
    if isinstance(obj, Pipeline):
        return obj.named_steps[list(obj.named_steps.keys())[-1]]
    return obj

def classes_of(obj):
    if hasattr(obj, "classes_"):
        return np.asarray(obj.classes_)
    est = final_estimator(obj)
    if hasattr(est, "classes_"):
        return np.asarray(est.classes_)
    raise ValueError("Candidate has no classes_")

def input_frame_for(obj):
    if hasattr(obj, "feature_names_in_"):
        cols = list(obj.feature_names_in_)
        if all(c in val.columns for c in cols):
            return val[cols]
        return None
    return val[raw_features]

def predict_proba_for(obj):
    frame = input_frame_for(obj)
    if frame is not None:
        try:
            proba = obj.predict_proba(frame)
            if len(proba) == len(val):
                return np.asarray(proba)
        except Exception:
            pass

    frontends = []
    if isinstance(pre, Pipeline) and isinstance(sel, Pipeline) and any(sel is s for s in pre.named_steps.values()):
        frontends.append([("pre", pre)])
    else:
        frontends.append([("pre", pre)])
        frontends.append([("pre", pre), ("sel", sel)])
        frontends.append([("sel", sel)])

    for steps in frontends:
        try:
            pipe = Pipeline(steps + [("model", obj)])
            first = steps[0][1]
            if hasattr(first, "feature_names_in_"):
                cols = list(first.feature_names_in_)
                if not all(c in val.columns for c in cols):
                    continue
                X = val[cols]
            else:
                X = val[raw_features]
            proba = pipe.predict_proba(X)
            if len(proba) == len(val):
                return np.asarray(proba)
        except Exception:
            continue
    raise RuntimeError("Could not obtain validation probabilities for a candidate")

def macro_ovr_auc(y_true, proba, classes):
    present = [c for c in classes if np.any(y_true == c)]
    aucs = []
    for c in present:
        j = int(np.where(classes == c)[0][0])
        y_bin = (y_true == c).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        aucs.append(roc_auc_score(y_bin, proba[:, j]))
    missing = [c for c in classes if not np.any(y_true == c)]
    if missing:
        print("Validation classes absent:", missing)
    return float(np.mean(aucs)) if aucs else float("nan")

def sig4(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    return float(f"{float(v):.4g}")

y_val = val[targets[0]].to_numpy()
rows = []
candidate_primary = {}
for name, obj in candidates:
    proba = predict_proba_for(obj)
    classes = classes_of(obj)
    score = macro_ovr_auc(y_val, proba, classes) if primary == "roc_auc_ovr" else float("nan")
    pred = classes[np.argmax(proba, axis=1)]
    macro_f1 = f1_score(y_val, pred, average="macro", labels=classes, zero_division=0)
    candidate_primary[name] = score
    rows.append({"model": name, "metric": primary, "value": sig4(score), "threshold": np.nan})
    rows.append({"model": name, "metric": "macro_f1", "value": sig4(macro_f1), "threshold": np.nan})
    mlca_runtime.record_metric(primary, float(score), "val")

lower_is_better = {"brier", "log_loss", "mae", "mse", "rmse", "mean_absolute_error", "mean_squared_error"}
if primary in lower_is_better:
    best_name = min(candidate_primary, key=lambda k: (np.isnan(candidate_primary[k]), candidate_primary[k]))
else:
    best_name = max(candidate_primary, key=lambda k: (np.isnan(candidate_primary[k]), candidate_primary[k]))
best_val = candidate_primary[best_name]

metrics_df = pd.DataFrame(rows)[["model", "metric", "value", "threshold"]]
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics by candidate model")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
if primary == "roc_auc_ovr":
    mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best_val), source="validation_metrics.csv")

print(f"train={train.shape} val={val.shape} features={len(raw_features)}")
print(f"{primary}: " + ", ".join(f"{n}={v:.4g}" for n, v in candidate_primary.items()))
print(f"best_model={best_name} best_{primary}={best_val:.4g}")
