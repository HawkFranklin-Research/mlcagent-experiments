import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
task = protocol["task"]
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
primary = task["primary_metric"]
if task.get("type") != "multiclass":
    raise ValueError("This script is for the declared multiclass task")
if len(targets) != 1:
    raise ValueError("Expected exactly one declared target")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
test = pd.read_parquet(inp["test"])
target = targets[0]

def key_like(s):
    if s.nunique(dropna=False) != len(s):
        return False
    if pd.api.types.is_float_dtype(s) or pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_integer_dtype(s):
        return False
    if pd.api.types.is_integer_dtype(s):
        return True
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)

drop = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in drop and not key_like(train[c])]

def load_pkl(key):
    return pickle.loads(inp[key].read_bytes())

pre = load_pkl("05_preprocess_fit_preprocessing_K97TX9BR/preprocessor.pkl")
sel = load_pkl("06_features_select_features_RVKX3HX3/selector.pkl")

def complete(model):
    if isinstance(model, Pipeline) and hasattr(model, "predict_proba"):
        return model
    steps = []
    if isinstance(sel, Pipeline):
        has_pre = any(("pre" in k.lower()) or isinstance(v, (ColumnTransformer, Pipeline)) for k, v in sel.named_steps.items())
        if not has_pre:
            steps.append(("pre", pre))
        steps.append(("selector", sel))
    else:
        steps.extend([("pre", pre), ("selector", sel)])
    steps.append(("model", model))
    return Pipeline(steps)

def expected_cols(m):
    names = getattr(m, "feature_names_in_", None)
    if names is None and isinstance(m, Pipeline):
        for step in m.named_steps.values():
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                break
    return list(names) if names is not None else None

def make_X(df, m):
    X = df[feature_cols].copy()
    exp = expected_cols(m)
    if exp is not None:
        missing = [c for c in exp if c not in X.columns]
        if missing:
            raise ValueError(f"Missing expected feature columns: {missing}")
        X = X[exp]
    return X

def classes_of(m):
    c = getattr(m, "classes_", None)
    if c is None and isinstance(m, Pipeline):
        c = list(m.named_steps.values())[-1].classes_
    return np.asarray(c)

def metric_value(name, y, proba, classes):
    y = np.asarray(y)
    if name == "roc_auc_ovr":
        present = [c for c in classes if c in set(y)]
        if len(present) < 2:
            raise ValueError("Too few classes in resample")
        idx = [int(np.where(classes == c)[0][0]) for c in present]
        return roc_auc_score(y, proba[:, idx], multi_class="ovr", average="macro", labels=present)
    if name in ("macro_f1", "f1"):
        yp = classes[np.argmax(proba, axis=1)]
        return f1_score(y, yp, average="macro", labels=classes, zero_division=0)
    if name == "roc_auc":
        return roc_auc_score(y, proba[:, 1])
    raise ValueError(f"Unsupported declared metric: {name}")

def boot(name, y, proba, classes, rng):
    vals = []
    for _ in range(1000):
        idx = rng.integers(0, len(y), len(y))
        try:
            vals.append(metric_value(name, y[idx], proba[idx], classes))
        except Exception:
            pass
    if not vals:
        return np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)

candidates = [
    ("logistic", complete(load_pkl("07_model_train_logistic_D2WYGTBM/model.pkl"))),
    ("alternative", complete(load_pkl("08_model_train_alternative_5QQJK8QV/alternative_model.pkl"))),
    ("latest", complete(load_pkl("latest_model"))),
]
yv = val[target].to_numpy()
direction = -1.0 if primary in ("brier_score", "log_loss", "rmse", "mae") else 1.0
best = None
for name, m in candidates:
    try:
        pv = m.predict_proba(make_X(val, m))
        score = direction * metric_value(primary, yv, pv, classes_of(m))
        if best is None or score > best[0]:
            best = (score, name, m)
    except Exception:
        continue
if best is None:
    raise ValueError("No candidate model could be scored on validation")
chosen_name, model = best[1], best[2]
classes = classes_of(model)
y = test[target].to_numpy()
proba = model.predict_proba(make_X(test, model))

p_val = metric_value(primary, y, proba, classes)
p_lo, p_hi = boot(primary, y, proba, classes, np.random.default_rng(42))
f_val = metric_value("macro_f1", y, proba, classes)
f_lo, f_hi = boot("macro_f1", y, proba, classes, np.random.default_rng(42))

out = pd.DataFrame([
    {"metric": primary, "value": p_val, "ci_low": p_lo, "ci_high": p_hi},
    {"metric": "macro_f1", "value": f_val, "ci_low": f_lo, "ci_high": f_hi},
])
def s4(x):
    return float(f"{x:.4g}") if pd.notna(x) else np.nan
for c in ["value", "ci_low", "ci_high"]:
    out[c] = out[c].map(s4)
mlca_runtime.save_table(out, "test_metrics.csv", caption="Frozen model test metrics with 95% bootstrap CI")

mlca_runtime.record_metric(primary, float(p_val), "test", ci=(float(p_lo), float(p_hi)))
mlca_runtime.record_metric("macro_f1", float(f_val), "test", ci=(float(f_lo), float(f_hi)))
if "roc_auc" in primary:
    mlca_runtime.record_fact_candidate("test_roc_auc", float(p_val), source="test_metrics.csv")

print("shapes", train.shape, val.shape, test.shape)
print("features", len(feature_cols))
print("selected", chosen_name, primary, round(float(p_val), 4))
