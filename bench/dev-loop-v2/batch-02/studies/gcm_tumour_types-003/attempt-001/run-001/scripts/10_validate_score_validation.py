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
        raise KeyError(f"Missing {k!r}. Available top-level keys: {list(protocol)}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
primary = protocol["task"]["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def is_key_like(s):
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=False) == len(s):
        name = str(s.name).lower()
        return any(k in name for k in ["id", "code", "index", "row"])
    return False

drop = set(targets) | set(excluded) | ({group} if group else set())
feat_cols = [c for c in train.columns if c not in drop and not is_key_like(train[c])]
target = targets[0]

def load_pkl(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("06_preprocess_fit_preprocessing_PDW4RHJ4/preprocessor.pkl")
sel = load_pkl("07_features_select_features_88EXJ3SM/selector.pkl")
models = {
    "08_model_train_logistic_SXT0E7SD": load_pkl("08_model_train_logistic_SXT0E7SD/model.pkl"),
    "09_model_train_alternative_H4Z5MZEV": load_pkl("09_model_train_alternative_H4Z5MZEV/alternative_model.pkl"),
}

def cols_of(obj):
    return list(obj.feature_names_in_) if hasattr(obj, "feature_names_in_") else None

def accepts_raw(obj):
    c = cols_of(obj)
    return c is not None and set(c).issubset(set(feat_cols))

def get_classes(obj):
    if hasattr(obj, "classes_"):
        return np.asarray(obj.classes_)
    if isinstance(obj, Pipeline):
        return np.asarray(obj.steps[-1][1].classes_)
    raise AttributeError("Model has no classes_")

def make_predictor(model_obj):
    if accepts_raw(model_obj):
        c = cols_of(model_obj)
        return lambda df: model_obj.predict_proba(df[c])
    if sel is not None and accepts_raw(sel):
        c = cols_of(sel)
        front = lambda df: sel.transform(df[c])
    elif pre is not None and accepts_raw(pre):
        cp = cols_of(pre)
        if sel is not None:
            front = lambda df: sel.transform(pre.transform(df[cp]))
        else:
            front = lambda df: pre.transform(df[cp])
    elif sel is not None:
        front = lambda df: sel.transform(df[feat_cols])
    elif pre is not None:
        front = lambda df: pre.transform(df[feat_cols])
    else:
        front = lambda df: df[feat_cols]
    return lambda df: model_obj.predict_proba(front(df))

y_val = val[target].to_numpy()
rows, results = [], []
for name, obj in models.items():
    proba = np.asarray(make_predictor(obj)(val))
    classes = get_classes(obj)
    if proba.ndim == 1:
        proba = proba[:, None]
    present = set(y_val)
    if len(classes) == 2:
        auc = roc_auc_score((y_val == classes[1]).astype(int), proba[:, 1])
    elif len(present) == len(classes):
        auc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)
    else:
        vals = [roc_auc_score((y_val == c).astype(int), proba[:, j])
                for j, c in enumerate(classes) if c in present]
        auc = float(np.mean(vals))
        print("missing classes:", sorted(set(classes) - present))
    pred = classes[np.argmax(proba, axis=1)]
    f1 = f1_score(y_val, pred, average="macro", labels=classes, zero_division=0)
    rows.append({"model": name, "metric": primary, "value": float(auc), "threshold": np.nan})
    rows.append({"model": name, "metric": "f1_macro", "value": float(f1), "threshold": np.nan})
    results.append((name, float(auc), float(f1)))

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
metrics_df["value"] = metrics_df["value"].map(lambda v: float(f"{v:.4g}"))
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics")

lower_is_better = primary in {"brier_score", "log_loss", "mse", "mae", "rmse"}
ordered = sorted(results, key=lambda x: x[1], reverse=lower_is_better)
for name, auc, _ in ordered:
    mlca_runtime.record_metric(primary, auc, "val")
best_name, best_val, _ = ordered[-1]
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_val, source="validation_metrics.csv")
print(f"train={train.shape} val={val.shape} features={len(feat_cols)}")
print(f"best_model={best_name} {primary}={best_val:.4f}")
