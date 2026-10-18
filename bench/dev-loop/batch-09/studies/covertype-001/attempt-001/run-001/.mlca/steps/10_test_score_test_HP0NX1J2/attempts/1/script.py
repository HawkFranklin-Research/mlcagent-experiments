import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key}; available top-level keys: {list(protocol)}")
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; task keys: {list(task)}")
primary_metric = task["primary_metric"]
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column", None)
if not targets:
    raise ValueError("No targets declared in protocol['targets']")

train = pd.read_parquet(inputs["train"])
test = pd.read_parquet(inputs["test"])
target = targets[0]

def is_keylike(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False

features = []
for col in train.columns:
    if col == target or col in excluded or (group is not None and col == group):
        continue
    if is_keylike(train[col]):
        continue
    features.append(col)

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_pkl(inputs["latest_model"])
if not isinstance(model, Pipeline):
    pre = load_pkl(inputs["05_preprocess_fit_preprocessing_0FJ6AMWD/preprocessor.pkl"])
    sel = load_pkl(inputs["06_features_select_features_79Q93RHQ/selector.pkl"])
    sel_steps = getattr(sel, "named_steps", {})
    wraps_pre = any("preprocess" in k.lower() for k in sel_steps)
    steps = [("selector", sel), ("model", model)] if wraps_pre else [
        ("preprocessor", pre), ("selector", sel), ("model", model)
    ]
    model = Pipeline(steps)

def first_input_columns(est):
    if isinstance(est, Pipeline):
        for _, step in est.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    if hasattr(est, "feature_names_in_"):
        return list(est.feature_names_in_)
    return None

expected = first_input_columns(model)
if expected is not None:
    missing = [c for c in expected if c not in test.columns]
    if missing:
        raise KeyError(f"Missing model input columns: {missing}")
    X_test = test[expected]
else:
    X_test = test[features]

y_test = test[target].to_numpy()
final = model
while isinstance(final, Pipeline):
    final = final.steps[-1][1]
classes = np.asarray(final.classes_)

proba = model.predict_proba(X_test)
pred = model.predict(X_test)

auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=classes)
f1 = f1_score(y_test, pred, average="macro", labels=classes)

rng = np.random.default_rng(42)
class_indices = [np.where(y_test == c)[0] for c in classes]
auc_vals, f1_vals = [], []
for _ in range(1000):
    idx = np.concatenate([rng.choice(ci, size=len(ci), replace=True) for ci in class_indices])
    yb, pb, db = y_test[idx], proba[idx], pred[idx]
    auc_vals.append(roc_auc_score(yb, pb, multi_class="ovr", average="macro", labels=classes))
    f1_vals.append(f1_score(yb, db, average="macro", labels=classes))
auc_ci = np.percentile(auc_vals, [2.5, 97.5])
f1_ci = np.percentile(f1_vals, [2.5, 97.5])

if primary_metric == "f1_macro":
    primary_value, primary_ci = f1, f1_ci
else:
    primary_value, primary_ci = auc, auc_ci

metrics = pd.DataFrame([
    {"metric": primary_metric, "value": primary_value, "ci_low": primary_ci[0], "ci_high": primary_ci[1]},
    {"metric": "f1_macro", "value": f1, "ci_low": f1_ci[0], "ci_high": f1_ci[1]},
]).round(4)
mlca_runtime.save_table(metrics, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")
mlca_runtime.record_metric(primary_metric, float(primary_value), "test", ci=(float(primary_ci[0]), float(primary_ci[1])))
mlca_runtime.record_metric("f1_macro", float(f1), "test", ci=(float(f1_ci[0]), float(f1_ci[1])))
mlca_runtime.record_fact_candidate("test_roc_auc", float(auc), source="test_metrics.csv")

print(f"features={len(features)} test_rows={len(test)}")
print(f"{primary_metric}={primary_value:.4f} CI=({primary_ci[0]:.4f}, {primary_ci[1]:.4f})")
