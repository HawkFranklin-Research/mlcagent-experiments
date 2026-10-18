import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing {key}; top-level keys: {list(protocol.keys())}")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; task keys: {list(task.keys())}")
primary = task["primary_metric"]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

test = pd.read_parquet(inp["test"])
vpath = inp["09_validate_score_validation_AE5E8TGS/validation_metrics.parquet"]
vm = pd.read_parquet(vpath) if vpath.suffix == ".parquet" else pd.read_csv(vpath)
lower = {str(c).lower(): c for c in vm.columns}
mcol = lower.get("model") or lower.get("candidate") or lower.get("model_name")
metcol = lower.get("metric") or lower.get("metric_name")
vcol = lower.get("value") or lower.get("score")
if not mcol or not metcol or not vcol:
    mcol, metcol, vcol = vm.columns[:3]
tcol = lower.get("target")

def val_score(model_name):
    sub = vm[vm[mcol].astype(str) == str(model_name)].copy()
    pm = sub[sub[metcol].astype(str).str.lower().str.contains(str(primary).lower(), regex=False)]
    if pm.empty:
        return np.nan
    names = pm[metcol].astype(str).str.lower()
    if tcol and tcol in pm.columns:
        tn = pm[tcol].astype(str).str.lower()
        macro = pm[tn.str.contains("macro|mean|avg|average", regex=True, na=False)]
        if len(macro):
            return float(pd.to_numeric(macro[vcol], errors="coerce").mean())
    macro = pm[names.str.contains("macro|mean|avg|average", regex=True)]
    if len(macro):
        return float(pd.to_numeric(macro[vcol], errors="coerce").mean())
    exact = pm[names == str(primary).lower()]
    if len(exact):
        return float(pd.to_numeric(exact[vcol], errors="coerce").mean())
    return float(pd.to_numeric(pm[vcol], errors="coerce").mean())

models = list(dict.fromkeys(vm[mcol].astype(str)))
scores = {m: val_score(m) for m in models}
scores = {m: s for m, s in scores.items() if pd.notna(s)}
if not scores:
    raise ValueError("No validation scores for declared primary metric")
higher_better = str(primary).lower() in {"r2", "roc_auc", "accuracy", "f1", "precision", "recall", "auc"}
selected_name = max(scores, key=scores.get) if higher_better else min(scores, key=scores.get)

def model_path(name):
    s = str(name).lower()
    if "baseline" in s or s in {"model", "model.pkl"}:
        return inp["07_model_train_baseline_WTZ11RHD/model.pkl"]
    if "alternative" in s or "alt" in s or "latest" in s:
        return inp["08_model_train_alternative_KHZYBTMD/alternative_model.pkl"]
    return None

sp = model_path(selected_name) or inp["latest_model"]
model = load_pkl(sp)
if not isinstance(model, Pipeline):
    pre = load_pkl(inp["05_preprocess_fit_preprocessing_9HEDMWQR/preprocessor.pkl"])
    sel = load_pkl(inp["06_features_retain_features_HD824TRJ/selector_all.pkl"])
    model = Pipeline([("preprocessor", pre), ("selector", sel), ("model", model)])

def key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return len(s) > 1 and s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return len(s) > 1 and s.nunique(dropna=False) == len(s)
    return False

base_excl = set(targets) | set(excluded) | ({group} if group else set())
key_cols = [c for c in test.columns if c not in base_excl and key_like(test[c])]
feat_cols = [c for c in test.columns if c not in base_excl and c not in set(key_cols)]
X_test = test[feat_cols]
if hasattr(model, "feature_names_in_"):
    names = [str(n) for n in model.feature_names_in_]
    if all(n in test.columns for n in names):
        X_test = test[names]

pred = np.asarray(model.predict(X_test))
y = test[targets].to_numpy()
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)

rows = []
for i, target in enumerate(targets):
    mse = mean_squared_error(y[:, i], pred[:, i])
    vals = {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y[:, i], pred[:, i])),
        "r2": float(r2_score(y[:, i], pred[:, i])),
    }
    for metric_name, value in vals.items():
        rows.append({"model": selected_name, "metric": f"{metric_name}_{target}", "value": value})
for metric_name in ["rmse", "mae", "r2"]:
    vals = [r["value"] for r in rows if r["metric"].startswith(metric_name + "_")]
    rows.append({"model": selected_name, "metric": metric_name, "value": float(np.mean(vals))})

def sig4(x):
    return float(f"{x:.4g}")

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value"])
metrics_df["value"] = metrics_df["value"].map(sig4)
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics for validation-selected candidate")

primary_row = metrics_df[metrics_df["metric"].astype(str).str.lower() == str(primary).lower()]
if primary_row.empty:
    primary_row = metrics_df[metrics_df["metric"].astype(str).str.lower().str.contains(str(primary).lower(), regex=False)]
primary_value = float(primary_row["value"].iloc[0])
for _, row in metrics_df.iterrows():
    mlca_runtime.record_metric(str(row["metric"]), float(row["value"]), split="test")
mlca_runtime.record_fact_candidate("test_score", primary_value, source="test_metrics.csv")
print("test rows", len(test))
print("selected", selected_name)
print(primary, primary_value)
