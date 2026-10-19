import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

if "task" not in protocol or "primary_metric" not in protocol["task"]:
    raise KeyError(f"Missing task.primary_metric; available top-level keys: {list(protocol)}")
primary = protocol["task"]["primary_metric"]
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")
if len(targets) != 1:
    raise ValueError(f"Expected exactly one target, got {targets}")
target = targets[0]

train = pd.read_parquet(inp["train"])
test = pd.read_parquet(inp["test"])
valm = pd.read_csv(inp["09_validate_score_validation_GN8A6T74/validation_metrics.csv"])

lower_better = any(k in primary.lower() for k in ("brier", "log_loss", "mse", "mae", "rmse"))
def best_name(df, metric_col, model_col):
    vals = pd.to_numeric(df[metric_col], errors="coerce")
    idx = vals.idxmin() if lower_better else vals.idxmax()
    return str(df.loc[idx, model_col]).lower()

model_path = inp["latest_model"]
model_col = next((c for c in valm.columns if c.lower() in ("model", "candidate", "model_name", "name")), None)
if primary in valm.columns and model_col is not None:
    name = best_name(valm, primary, model_col)
    if "logistic" in name:
        model_path = inp["07_model_train_logistic_H5RFG7PA/model.pkl"]
    elif "alternative" in name or "hist" in name:
        model_path = inp["08_model_train_alternative_5HD55PJ2/alternative_model.pkl"]
elif {"metric", "value"}.issubset(valm.columns):
    sub = valm[valm["metric"].astype(str) == primary]
    if len(sub):
        if model_col is None:
            model_col = next((c for c in sub.columns if c.lower() in ("model", "candidate", "model_name", "name")), None)
        if model_col is not None:
            name = best_name(sub, "value", model_col)
            if "logistic" in name:
                model_path = inp["07_model_train_logistic_H5RFG7PA/model.pkl"]
            elif "alternative" in name or "hist" in name:
                model_path = inp["08_model_train_alternative_5HD55PJ2/alternative_model.pkl"]

base_cols = [c for c in train.columns if c not in set(targets) | set(excluded) | ({group} if group else set())]
key_cols = []
for c in base_cols:
    s = train[c]
    nm = c.lower()
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=True) == len(s):
            key_cols.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s) and any(k in nm for k in ("id", "key", "code", "index")):
        key_cols.append(c)
feature_cols = [c for c in base_cols if c not in key_cols]

with open(model_path, "rb") as f:
    model = pickle.load(f)
model_cols = list(getattr(model, "feature_names_in_", feature_cols))
bad = set(model_cols) & (set(targets) | set(excluded))
if bad:
    raise ValueError(f"Model input columns include target/excluded columns: {sorted(bad)}")

X_test = test[model_cols]
y_true = test[target].to_numpy()
proba = model.predict_proba(X_test)
classes = np.asarray(model.classes_)

def macro_ovr_auc(y, p, cls):
    present = [c for c in cls if np.any(y == c)]
    if len(present) == len(cls):
        return float(roc_auc_score(y, p, multi_class="ovr", average="macro", labels=cls))
    vals = []
    for c in present:
        j = list(cls).index(c)
        yb = (y == c).astype(int)
        if len(np.unique(yb)) == 2:
            vals.append(roc_auc_score(yb, p[:, j]))
    return float(np.mean(vals)) if vals else float("nan")

auc = macro_ovr_auc(y_true, proba, classes)
pred = classes[np.argmax(proba, axis=1)]
f1 = f1_score(y_true, pred, average="macro", labels=classes, zero_division=0)

rng = np.random.default_rng(42)
n = len(y_true)
aucs, f1s = [], []
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yb, pb, predb = y_true[idx], proba[idx], pred[idx]
    aucs.append(macro_ovr_auc(yb, pb, classes))
    f1s.append(f1_score(yb, predb, average="macro", labels=classes, zero_division=0))
auc_ci = np.nanpercentile(aucs, [2.5, 97.5])
f1_ci = np.nanpercentile(f1s, [2.5, 97.5])

def r4(x):
    return float(f"{x:.4g}") if np.isfinite(x) else np.nan

primary_value = auc if primary.startswith("roc_auc") else f1 if primary.startswith("f1") else auc
metrics = pd.DataFrame([
    {"metric": primary, "value": r4(primary_value), "ci_low": r4(auc_ci[0] if primary.startswith("roc_auc") else f1_ci[0]), "ci_high": r4(auc_ci[1] if primary.startswith("roc_auc") else f1_ci[1])},
    {"metric": "f1_macro", "value": r4(f1), "ci_low": r4(f1_ci[0]), "ci_high": r4(f1_ci[1])},
], columns=["metric", "value", "ci_low", "ci_high"])

mlca_runtime.save_table(metrics, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI.")
mlca_runtime.record_metric(primary, float(primary_value), "test", ci=(float(auc_ci[0] if primary.startswith("roc_auc") else f1_ci[0]), float(auc_ci[1] if primary.startswith("roc_auc") else f1_ci[1])))
mlca_runtime.record_metric("f1_macro", float(f1), "test", ci=(float(f1_ci[0]), float(f1_ci[1])))
mlca_runtime.record_fact_candidate("test_roc_auc", float(auc), source="test_metrics.csv")

print(f"test shape={test.shape}; model={model_path.name}")
print(f"{primary}={primary_value:.4f} CI=({auc_ci[0]:.4f},{auc_ci[1]:.4f})")
print(f"f1_macro={f1:.4f} CI=({f1_ci[0]:.4f},{f1_ci[1]:.4f})")
