import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
missing = [k for k in ["targets", "excluded_columns", "group_column", "task"] if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol.keys())}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]
primary = protocol["task"]["primary_metric"]

train = pd.read_parquet(inputs["train"])
test = pd.read_parquet(inputs["test"])

def is_key_like(s):
    if s.nunique(dropna=False) != len(s):
        return False
    if pd.api.types.is_float_dtype(s):
        return False
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
        or pd.api.types.is_integer_dtype(s)
    )

drop = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    if is_key_like(train[c]):
        continue
    feature_cols.append(c)
if not feature_cols:
    raise ValueError("No feature columns remain after exclusions")
X_test = test.loc[:, feature_cols].copy()

valm = pd.read_csv(inputs["09_validate_score_validation_6A3SAZRV/validation_metrics.csv"])
if not {"model", "metric", "value"}.issubset(valm.columns):
    raise KeyError(f"validation_metrics.csv columns: {list(valm.columns)}")
cand = valm[valm["metric"].astype(str) == primary].copy()
cand["value"] = pd.to_numeric(cand["value"], errors="coerce")
cand = cand.dropna(subset=["value"])
if cand.empty:
    raise ValueError(f"No validation rows for primary metric {primary}")
best = cand.loc[cand["value"].idxmax()]
best_name = str(best["model"])
best_lower = best_name.lower()
model_path = inputs["08_model_train_alternative_CZ7QSYZK/alternative_model.pkl"] if any(
    x in best_lower for x in ["alt", "alternative", "latest"]
) else inputs["07_model_train_baseline_JP4JWWDQ/model.pkl"]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model_obj = load_pkl(model_path)

def make_predictor(model):
    if isinstance(model, Pipeline):
        steps = list(model.named_steps)
        if len(steps) > 1 or any(
            ("pre" in s.lower() or "select" in s.lower() or "feature" in s.lower())
            for s in steps
        ):
            return model
    pre = load_pkl(inputs["05_preprocess_fit_preprocessing_A8X1QVXT/preprocessor.pkl"])
    sel = load_pkl(inputs["06_features_retain_features_ZGB5PBQC/selector_all.pkl"])
    if isinstance(sel, Pipeline) and any(
        ("pre" in k.lower() or "preprocess" in k.lower()) for k in sel.named_steps
    ):
        return Pipeline([("selector", sel), ("model", model)])
    return Pipeline([("preprocessor", pre), ("selector", sel), ("model", model)])

predictor = make_predictor(model_obj)
pred = predictor.predict(X_test)
pred_arr = np.asarray(pred)
if pred_arr.ndim == 1:
    pred_df = pd.DataFrame({targets[0]: pred_arr}, index=test.index)
else:
    pred_df = pd.DataFrame(pred_arr, columns=targets, index=test.index)
y_true = test.loc[:, targets].copy()

f1s = {}
for t in targets:
    f1s[t] = float(f1_score(y_true[t], pred_df[t], average="macro", zero_division=0))
primary_val = float(np.mean(list(f1s.values())))
hamm = float(np.mean([np.mean(y_true[t].to_numpy() != pred_df[t].to_numpy()) for t in targets]))
subset = float(np.mean(np.all(y_true.to_numpy() == pred_df.to_numpy(), axis=1)))

rows = [{"model": best_name, "metric": f"macro_f1_{t}", "value": f1s[t]} for t in targets]
rows += [
    {"model": best_name, "metric": primary, "value": primary_val},
    {"model": best_name, "metric": "hamming_loss", "value": hamm},
    {"model": best_name, "metric": "subset_accuracy", "value": subset},
]
metrics_df = pd.DataFrame(rows)
metrics_df["value"] = metrics_df["value"].map(lambda x: float(f"{x:.4g}"))
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Frozen validation-selected candidate test metrics")
mlca_runtime.record_metric(primary, primary_val, "test")
mlca_runtime.record_fact_candidate("test_score", primary_val, source="test_metrics.csv")

print(f"selected={best_name} features={len(feature_cols)} test_rows={len(test)}")
print(f"{primary}={primary_val:.4f} hamming={hamm:.4f} subset={subset:.4f}")
