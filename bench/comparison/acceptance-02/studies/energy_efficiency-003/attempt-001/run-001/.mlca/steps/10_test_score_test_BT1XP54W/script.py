import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out_dir = mlca_runtime.output_dir()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")
task = protocol["task"]
primary = task["primary_metric"]
lower_better = primary.lower() not in {"r2", "r2_score", "auc", "roc_auc", "accuracy", "f1", "f1_score"}

def is_keylike(s: pd.Series) -> bool:
    n = len(s)
    if n == 0:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == n
    if pd.api.types.is_integer_dtype(s):
        return bool(s.is_unique and s.nunique(dropna=False) == n)
    return False

train = pd.read_parquet(inputs["train"])
base_cols = [c for c in train.columns if c not in set(targets + excluded + ([group] if group else []))]
feature_cols = [c for c in base_cols if not is_keylike(train[c])]

vm = pd.read_csv(inputs["09_validate_score_validation_F12RYJBQ/validation_metrics.csv"])
model_col = next((c for c in vm.columns if "model" in c.lower()), None)
metric_col = next((c for c in vm.columns if "metric" in c.lower()), None)
value_col = next((c for c in vm.columns if "value" in c.lower() or c.lower() == "score"), None)
if not model_col or not metric_col or not value_col:
    raise ValueError(f"Validation metrics columns are {list(vm.columns)}")

v = vm[vm[metric_col].astype(str).str.lower() == str(primary).lower()]
if v.empty:
    v = vm[vm[metric_col].astype(str).str.lower().str.contains(str(primary).lower(), regex=False)]
if v.empty:
    raise ValueError(f"Primary metric {primary!r} not found in validation metrics")
target_col = next((c for c in v.columns if "target" in c.lower()), None)
if target_col is not None:
    mask = v[target_col].astype(str).isin([str(t) for t in targets])
    if mask.any():
        v = v[mask]
agg = v.groupby(model_col)[value_col].mean()
selected_model = str(agg.idxmin() if lower_better else agg.idxmax())

sl = selected_model.lower()
if "alternative" in sl or "latest" in sl:
    model_path = inputs["08_model_train_alternative_QSEDYM24/alternative_model.pkl"]
elif "baseline" in sl:
    model_path = inputs["07_model_train_baseline_21R1ZBKQ/model.pkl"]
else:
    raise ValueError(f"Cannot map selected model {selected_model!r} to a saved candidate")

with open(model_path, "rb") as f:
    model = pickle.load(f)
if isinstance(model, dict) and "model" in model:
    model = model["model"]

test = pd.read_parquet(inputs["test"])
X_test = test[feature_cols]
y_test = test[targets]

def predict_model(model, X):
    try:
        return model.predict(X)
    except Exception:
        pre_path = inputs["05_preprocess_fit_preprocessing_H2H2H4ZR/preprocessor.pkl"]
        sel_path = inputs["06_features_retain_features_3E3KFT0A/selector_all.pkl"]
        with open(pre_path, "rb") as f:
            pre = pickle.load(f)
        with open(sel_path, "rb") as f:
            sel = pickle.load(f)
        if isinstance(pre, dict) and "model" in pre:
            pre = pre["model"]
        if isinstance(sel, dict) and "model" in sel:
            sel = sel["model"]

        Z = X
        if hasattr(sel, "feature_names_in_") and list(sel.feature_names_in_) == list(Z.columns):
            Z = sel.transform(Z)
        else:
            if hasattr(pre, "feature_names_in_") and list(pre.feature_names_in_) == list(Z.columns):
                Z = pre.transform(Z)
                if not isinstance(Z, pd.DataFrame):
                    names = pre.get_feature_names_out() if hasattr(pre, "get_feature_names_out") else None
                    if names is not None:
                        Z = pd.DataFrame(Z, columns=names, index=X.index)
            if hasattr(sel, "feature_names_in_") and list(sel.feature_names_in_) == list(Z.columns):
                Z = sel.transform(Z)
        return model.predict(Z)

pred = np.asarray(predict_model(model, X_test))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
if pred.shape[1] != len(targets):
    raise ValueError(f"Prediction shape {pred.shape} does not match targets {targets}")

rows = []
metric_values = {"rmse": [], "mae": [], "r2": []}
for j, target in enumerate(targets):
    yt = y_test[target].to_numpy()
    yp = pred[:, j]
    vals = {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
    }
    for metric, val in vals.items():
        metric_values[metric].append(val)
        rows.append({"model": selected_model, "metric": f"{metric}_{target}", "value": val})

macro_primary = float(np.mean(metric_values[primary] if primary in metric_values else metric_values["rmse"]))
for metric, vals in metric_values.items():
    macro = float(np.mean(vals))
    metric_name = primary if metric == primary else f"{metric}_macro"
    rows.append({"model": selected_model, "metric": metric_name, "value": macro})

def sig4(x):
    return float(f"{x:.4g}") if np.isfinite(x) else x

metrics_df = pd.DataFrame(rows)
metrics_df["value"] = metrics_df["value"].map(sig4)
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics for validation-selected candidate")

mlca_runtime.record_metric(primary, macro_primary, "test")
mlca_runtime.record_fact_candidate("test_score", macro_primary, source="test_metrics.csv")

print(f"selected={selected_model}")
print(f"test_rows={len(test)}")
print(f"{primary}_macro={macro_primary:.4g}")
