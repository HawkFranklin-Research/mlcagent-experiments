import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
task = protocol["task"]
primary = task["primary_metric"]
targets = protocol["targets"]
targets = [targets] if isinstance(targets, str) else list(targets)
excluded = protocol["excluded_columns"]
excluded = [excluded] if isinstance(excluded, str) else list(excluded)
group = protocol.get("group_column")
target_col = targets[0]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def is_key_like(s, n):
    if n <= 1 or s.nunique(dropna=True) != n:
        return False
    return (pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_integer_dtype(s.dtype)
            or pd.api.types.is_bool_dtype(s.dtype))

def feature_columns(df):
    cand = [c for c in df.columns if c not in set(targets) | set(excluded) and c != group]
    return [c for c in cand if not is_key_like(df[c], len(df))]

feature_cols = feature_columns(train)

def model_proba(model, df):
    cols = getattr(model, "feature_names_in_", None)
    cols = list(cols) if cols is not None else feature_cols
    proba = model.predict_proba(df.loc[:, cols])
    classes = getattr(model, "classes_", None)
    return proba, np.asarray(classes)

def macro_ovr_auc(y, proba, classes):
    present = [c for c in classes if (y == c).any()]
    missing = [c for c in classes if c not in present]
    if len(present) == len(classes):
        return roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes), missing
    vals = []
    for c in present:
        j = int(np.where(classes == c)[0][0])
        vals.append(roc_auc_score((y == c).astype(int), proba[:, j]))
    return float(np.mean(vals)), missing

def sig(x):
    return float(f"{float(x):.4g}")

model_paths = {
    "logistic": inp["08_model_train_logistic_DZS6CMQH/model.pkl"],
    "alternative": inp["09_model_train_alternative_SHFBCB82/alternative_model.pkl"],
}

rows, results = [], []
y_val = val[target_col]
for name, path in model_paths.items():
    model = load_pkl(path)
    proba, classes = model_proba(model, val)
    auc, missing = macro_ovr_auc(y_val, proba, classes)
    pred = classes[np.argmax(proba, axis=1)]
    f1 = f1_score(y_val, pred, labels=classes, average="macro", zero_division=0)
    rows.append({"model": name, "metric": primary, "value": sig(auc), "threshold": np.nan})
    rows.append({"model": name, "metric": "macro_f1", "value": sig(f1), "threshold": np.nan})
    mlca_runtime.record_metric(primary, float(auc), "val")
    results.append((name, float(auc)))
    if missing:
        print(f"{name} missing val classes: {missing}")

lower_better = {"brier_score", "log_loss", "mse", "mae", "rmse"}
best_name, best_val = (min(results, key=lambda x: x[1]) if primary in lower_better
                       else max(results, key=lambda x: x[1]))

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics by candidate model")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
if primary == "roc_auc_ovr":
    mlca_runtime.record_fact_candidate("best_val_roc_auc", best_val, source="validation_metrics.csv")

print("train", train.shape, "val", val.shape)
print("features", len(feature_cols))
print("best", best_name, sig(best_val), "primary", primary)
