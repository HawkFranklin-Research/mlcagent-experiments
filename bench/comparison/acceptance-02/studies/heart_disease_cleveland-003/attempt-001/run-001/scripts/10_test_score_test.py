import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import f1_score, hamming_loss
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in proto:
        raise KeyError(f"Missing '{key}' in _protocol; available keys: {list(proto.keys())}")
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
task = proto["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing 'primary_metric' in _protocol['task']; available keys: {list(task.keys())}")
primary = task["primary_metric"]
group = proto.get("group_column", None)

val = pd.read_csv(inputs["09_validate_score_validation_QG0J48N2/validation_metrics.csv"])
cols = {c.lower(): c for c in val.columns}
if {"model", "metric", "value"} <= set(cols):
    mcol, metcol, vcol = cols["model"], cols["metric"], cols["value"]
    rows = val[val[metcol].astype(str).str.lower() == str(primary).lower()]
    if rows.empty:
        rows = val[val[metcol].astype(str).str.lower().str.contains(str(primary).lower(), regex=False)]
    if rows.empty:
        raise ValueError(f"No validation rows for primary metric {primary!r}; columns={list(val.columns)}")
    best_model = rows.groupby(mcol, as_index=False)[vcol].max().sort_values(vcol, ascending=False).iloc[0][mcol]
elif "model" in cols and str(primary) in cols:
    best_model = val.sort_values(cols[str(primary)], ascending=False).iloc[0][cols["model"]]
else:
    raise ValueError(f"Unrecognized validation metrics format; columns={list(val.columns)}")

name = str(best_model).lower()
if "alt" in name or "08" in name:
    model_path = inputs["08_model_train_alternative_5NAZBJJ0/alternative_model.pkl"]
elif "base" in name or "07" in name:
    model_path = inputs["07_model_train_baseline_X4GJNHE4/model.pkl"]
else:
    raise ValueError(f"Cannot map validation-selected model {best_model!r} to a declared candidate path")

with open(model_path, "rb") as f:
    model = pickle.load(f)

if isinstance(model, Pipeline):
    predictor = model
else:
    with open(inputs["06_features_retain_features_NH52E6T3/selector_all.pkl"], "rb") as f:
        selector = pickle.load(f)
    with open(inputs["05_preprocess_fit_preprocessing_DXHK0F72/preprocessor.pkl"], "rb") as f:
        preprocessor = pickle.load(f)
    if isinstance(selector, Pipeline) or hasattr(selector, "feature_names_in_"):
        predictor = Pipeline([("selector", selector), ("model", model)])
    else:
        predictor = Pipeline([("preprocessor", preprocessor), ("selector", selector), ("model", model)])

def expected_features(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        first = obj.named_steps[list(obj.named_steps)[0]]
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
    train = pd.read_parquet(inputs["train"])
    return [c for c in train.columns if c not in targets and c not in excluded and c != group]

features = expected_features(predictor)
test = pd.read_parquet(inputs["test"])
missing = [c for c in features if c not in test.columns]
if missing:
    raise KeyError(f"Missing expected test feature columns: {missing[:10]}")
X_test = test[features].copy()

y_pred = predictor.predict(X_test)
if isinstance(y_pred, list):
    y_pred = np.column_stack([np.asarray(p).ravel() for p in y_pred])
else:
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1) if len(targets) == 1 else y_pred.reshape(-1, len(targets))
    if y_pred.ndim == 2 and y_pred.shape[1] != len(targets) and y_pred.shape[0] == len(targets):
        y_pred = y_pred.T

def final_est(m):
    return m.named_steps[list(m.named_steps)[-1]] if isinstance(m, Pipeline) else m

def classes_for(est, i):
    if hasattr(est, "estimators_"):
        try:
            return getattr(est.estimators_[i], "classes_", None)
        except Exception:
            return None
    c = getattr(est, "classes_", None)
    if isinstance(c, (list, tuple)) and len(c) == len(targets):
        return c[i]
    return c

est = final_est(predictor)
f1s, hams = [], []
all_match = np.ones(len(test), dtype=bool)
for i, t in enumerate(targets):
    yt = test[t].to_numpy()
    yp = y_pred[:, i]
    cls = classes_for(est, i)
    if cls is not None:
        cls = np.asarray(cls)
        ypa = np.asarray(yp)
        if ypa.dtype.kind in "iu" and np.all((ypa >= 0) & (ypa < len(cls))):
            yp = cls[ypa]
    f1s.append(float(f1_score(yt, yp, average="macro", zero_division=0)))
    hams.append(float(hamming_loss(yt, yp)))
    all_match &= (yt == yp)

macro_f1 = float(np.mean(f1s))
hamming_avg = float(np.mean(hams))
subset_acc = float(np.mean(all_match))

rows = [{"model": best_model, "metric": str(primary), "value": macro_f1}]
rows += [{"model": best_model, "metric": f"macro_f1_{t}", "value": v} for t, v in zip(targets, f1s)]
rows += [{"model": best_model, "metric": f"hamming_loss_{t}", "value": v} for t, v in zip(targets, hams)]
rows += [
    {"model": best_model, "metric": "hamming_loss_avg", "value": hamming_avg},
    {"model": best_model, "metric": "subset_accuracy", "value": subset_acc},
]
out = pd.DataFrame(rows)
out["value"] = pd.to_numeric(out["value"]).round(4)
mlca_runtime.save_table(out, "test_metrics.csv", caption="Test metrics for frozen validation-selected model")
mlca_runtime.record_metric(str(primary), macro_f1, "test")
mlca_runtime.record_fact_candidate("test_score", macro_f1, source="test_metrics.csv")

print(f"model={best_model} n_test={len(test)}")
print(f"{primary}={macro_f1:.4f} hamming_avg={hamming_avg:.4f} subset_acc={subset_acc:.4f}")
