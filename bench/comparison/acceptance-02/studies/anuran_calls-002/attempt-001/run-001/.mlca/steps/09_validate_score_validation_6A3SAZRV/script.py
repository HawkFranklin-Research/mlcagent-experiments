import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, hamming_loss

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for k in ("targets", "excluded_columns", "group_column", "task"):
    if k not in protocol:
        raise KeyError(f"Missing protocol key {k!r}; available top-level keys: {list(protocol)}")
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; task keys: {list(task)}")

targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group_col = protocol["group_column"]
primary = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_keylike(s):
    if pd.api.types.is_float_dtype(s):
        return False
    allowed = (pd.api.types.is_integer_dtype(s) or pd.api.types.is_object_dtype(s)
               or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype))
    return allowed and s.nunique(dropna=False) == len(s)

features = [c for c in train.columns
            if c not in set(targets) | excluded | {group_col} and not is_keylike(train[c])]
X_val = val[features].copy()
y_val = val[targets].copy()
y_true = y_val[targets].to_numpy()

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inputs["05_preprocess_fit_preprocessing_A8X1QVXT/preprocessor.pkl"])
sel = load_pkl(inputs["06_features_retain_features_ZGB5PBQC/selector_all.pkl"])

def unwrap(o):
    if isinstance(o, dict):
        for k in ("model", "pipeline", "estimator", "best_model"):
            if k in o:
                return o[k]
    return getattr(o, "best_estimator_", o)

def fni(o):
    f = getattr(o, "feature_names_in_", None)
    if f is None and isinstance(o, Pipeline):
        for s in o.named_steps.values():
            f = getattr(s, "feature_names_in_", None)
            if f is not None:
                break
    return None if f is None else list(f)

def has_selector(o):
    if not isinstance(o, Pipeline):
        return False
    return any(hasattr(s, "get_support") or (isinstance(s, Pipeline) and has_selector(s))
               for s in o.named_steps.values())

def frontend(X):
    sf = fni(sel)
    if sf is not None and set(sf) == set(features):
        return sel.transform(X)
    Xt = pre.transform(X) if pre is not None else X
    return sel.transform(Xt) if sel is not None else Xt

def predict_obj(o, X):
    o = unwrap(o)
    f = fni(o)
    if f is not None and set(f) == set(features):
        return np.asarray(o.predict(X))
    if f is None and not isinstance(o, Pipeline):
        try:
            return np.asarray(o.predict(X))
        except Exception:
            pass
    if isinstance(o, Pipeline) and has_selector(o):
        Xt = pre.transform(X) if pre is not None else X
        return np.asarray(o.predict(Xt))
    return np.asarray(o.predict(frontend(X)))

def to_matrix(pred, n, t):
    pred = np.asarray(pred)
    if pred.ndim == 1:
        if t == 1:
            return pred.reshape(-1, 1)
        if pred.dtype == object and len(pred) and isinstance(pred[0], (list, tuple, np.ndarray)):
            pred = np.vstack(pred)
        else:
            raise ValueError(f"Cannot reshape predictions {pred.shape} for {t} targets")
    if pred.ndim == 2 and pred.shape[1] == t:
        return pred
    if pred.ndim == 2 and pred.shape[0] == t:
        return pred.T
    raise ValueError(f"Unexpected prediction shape {pred.shape} for {t} targets")

candidates = [
    ("baseline", inputs["07_model_train_baseline_JP4JWWDQ/model.pkl"]),
    ("alternative", inputs["08_model_train_alternative_CZ7QSYZK/alternative_model.pkl"]),
]
rows = []
macro_scores = {}
primary_scores = {}
for name, path in candidates:
    model = load_pkl(path)
    pred = to_matrix(predict_obj(model, X_val), len(y_val), len(targets))
    f1s, hls = [], []
    for j, target in enumerate(targets):
        a, b = y_true[:, j], pred[:, j]
        f = f1_score(a, b, average="macro", zero_division=0)
        h = hamming_loss(a, b)
        f1s.append(f)
        hls.append(h)
        rows.append((name, f"macro_f1_{target}", float(f"{f:.4g}")))
        rows.append((name, f"hamming_loss_{target}", float(f"{h:.4g}")))
    macro = float(np.mean(f1s))
    ham = float(np.mean(hls))
    subset = float(np.mean(np.all(pred == y_true, axis=1)))
    macro_scores[name] = macro
    primary_scores[name] = macro
    rows.append((name, primary, float(f"{macro:.4g}")))
    rows.append((name, "hamming_loss", float(f"{ham:.4g}")))
    rows.append((name, "subset_accuracy", float(f"{subset:.4g}")))

metrics = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(metrics, "validation_metrics.csv", caption="Validation metrics")
best = max(macro_scores, key=macro_scores.get)
best_primary = primary_scores[best]
mlca_runtime.record_metric(primary, best_primary, "val")
mlca_runtime.record_fact_candidate("best_validation_score", best_primary, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best, source="validation_metrics.csv")
print(f"train={train.shape} val={val.shape} features={len(features)}")
print(f"best={best} {primary}={best_primary:.4g}")
