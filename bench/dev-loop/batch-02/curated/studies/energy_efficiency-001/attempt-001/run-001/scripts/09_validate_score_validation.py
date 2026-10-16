import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = proto["targets"]
excluded = proto.get("excluded_columns", [])
group = proto.get("group_column")
task = proto["task"]
primary = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

feat_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_integer_dtype(s) and s.nunique() == len(s) and len(s) > 1:
            continue
    else:
        if s.nunique() == len(s) and len(s) > 1:
            continue
    feat_cols.append(c)

X_val = val[feat_cols]
y_val = val[targets]

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load(inputs["05_preprocess_fit_preprocessing_K5VFSMR2/preprocessor.pkl"])
sel = load(inputs["06_features_retain_features_8F2KVPF9/selector_all.pkl"])
m_base = load(inputs["07_model_train_baseline_0WM3R8E0/model.pkl"])
m_alt = load(inputs["08_model_train_alternative_B8FQJ4R0/alternative_model.pkl"])

def build(model):
    if isinstance(model, Pipeline):
        names = [n.lower() for n in model.named_steps]
        if any("pre" in n or "select" in n for n in names):
            return model
    steps = []
    pre_has_sel = False
    if isinstance(pre, Pipeline):
        steps.extend(list(pre.steps))
        pre_has_sel = any("select" in n.lower() for n in pre.named_steps)
    else:
        steps.append(("pre", pre))
    if not pre_has_sel:
        if isinstance(sel, Pipeline):
            steps.extend(list(sel.steps))
        else:
            steps.append(("sel", sel))
    if isinstance(model, Pipeline):
        steps.extend(list(model.steps))
    else:
        steps.append(("model", model))
    return Pipeline(steps)

candidates = {"baseline": build(m_base), "alternative": build(m_alt)}

rows = []
scores = {}
for name, pipe in candidates.items():
    pred = np.asarray(pipe.predict(X_val))
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    rmses = []
    for i, t in enumerate(targets):
        yt = y_val[t].values
        pt = pred[:, i]
        rmse = float(np.sqrt(mean_squared_error(yt, pt)))
        mae = float(mean_absolute_error(yt, pt))
        r2 = float(r2_score(yt, pt))
        rows.append({"model": name, "metric": f"rmse_{t}", "value": rmse})
        rows.append({"model": name, "metric": f"mae_{t}", "value": mae})
        rows.append({"model": name, "metric": f"r2_{t}", "value": r2})
        rmses.append(rmse)
    macro = float(np.mean(rmses))
    rows.append({"model": name, "metric": "rmse_macro", "value": macro})
    scores[name] = macro

best = min(scores, key=scores.get)
best_score = scores[best]

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "validation_metrics.csv", "Validation metrics per model and target")

mlca_runtime.record_metric("rmse", best_score, "val")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best, source="validation_metrics.csv")

print(f"val rows: {len(val)}, features: {len(feat_cols)}")
print(f"best model: {best}, macro rmse: {best_score:.4f}")
