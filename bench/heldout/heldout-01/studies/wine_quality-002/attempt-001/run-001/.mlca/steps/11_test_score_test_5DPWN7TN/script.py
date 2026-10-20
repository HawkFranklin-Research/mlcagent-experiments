import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for k in ("task", "targets", "excluded_columns"):
    if k not in protocol:
        raise KeyError(f"Missing protocol key {k}; available: {list(protocol.keys())}")
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; available task keys: {list(task.keys())}")
targets = protocol["targets"]
target = targets[0]
primary = task["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
test = pd.read_parquet(inp["test"])

def load_obj(p):
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        import joblib
        return joblib.load(p)

pre = load_obj(inp["06_preprocess_fit_preprocessing_2FS9KME6/preprocessor.pkl"])
sel = load_obj(inp["07_features_retain_features_SY1WG08Y/selector_all.pkl"])
m08 = load_obj(inp["08_model_train_logistic_9FB709K5/model.pkl"])
m09 = load_obj(inp["latest_model"])

raw_cols = getattr(pre, "feature_names_in_", None)
if raw_cols is None:
    raw_cols = getattr(sel, "feature_names_in_", None)
if raw_cols is not None:
    raw_cols = list(raw_cols)

def assemble(model):
    if isinstance(model, Pipeline):
        fi = getattr(model, "feature_names_in_", None)
        if raw_cols is not None and fi is not None and list(fi) == raw_cols:
            return model
        if any(("preprocess" in k.lower() or "column" in k.lower() or "feature" in k.lower()) for k in model.named_steps):
            return model
    if isinstance(pre, Pipeline) and any("selector" in k.lower() for k in pre.named_steps):
        front = pre
    elif isinstance(sel, Pipeline) and any("preprocess" in k.lower() for k in sel.named_steps):
        front = sel
    else:
        front = Pipeline([("preprocessor", pre), ("selector", sel)])
    if isinstance(model, Pipeline):
        return Pipeline([("frontend", front)] + list(model.steps))
    return Pipeline([("frontend", front), ("model", model)])

def macro_ovr(y, proba, classes):
    vals = []
    for j, c in enumerate(classes):
        yb = (y == c).astype(int)
        if len(np.unique(yb)) == 2:
            vals.append(roc_auc_score(yb, proba[:, j]))
    return float(np.mean(vals)) if vals else np.nan

def derive_features(df):
    excl = set(targets) | set(protocol.get("excluded_columns", []))
    g = protocol.get("group_column")
    if g:
        excl.add(g)
    cols = []
    for c in df.columns:
        if c in excl:
            continue
        s = df[c]
        if (s.dtype == object or str(s.dtype).startswith("string") or isinstance(s.dtype, pd.CategoricalDtype)) and s.nunique(dropna=False) == len(df):
            continue
        cols.append(c)
    return cols

def score_model(model):
    p = assemble(model)
    cols = getattr(p, "feature_names_in_", None)
    if cols is None:
        cols = raw_cols
    if cols is None:
        cols = derive_features(train)
    cols = list(cols)
    proba = p.predict_proba(val[cols])
    return macro_ovr(val[target].to_numpy(), proba, list(p.classes_)), p, cols

s08, p08, c08 = score_model(m08)
s09, p09, c09 = score_model(m09)
if (s08 if np.isfinite(s08) else -np.inf) >= (s09 if np.isfinite(s09) else -np.inf):
    predictor, cols, selected, best = p08, c08, "08_logistic", s08
else:
    predictor, cols, selected, best = p09, c09, "09_alternative", s09

y_test = test[target].to_numpy()
X_test = test[cols]
proba = predictor.predict_proba(X_test)
classes = list(predictor.classes_)
y_pred = predictor.predict(X_test)
value = macro_ovr(y_test, proba, classes)

def boot_ci(y, pb, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    nrows = len(y)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, nrows, nrows)
        v = macro_ovr(y[idx], pb[idx], classes)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

def boot_f1(y, yp, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    nrows = len(y)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, nrows, nrows)
        vals.append(f1_score(y[idx], yp[idx], labels=classes, average="macro", zero_division=0))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

lo, hi = boot_ci(y_test, proba)
f1v = f1_score(y_test, y_pred, labels=classes, average="macro", zero_division=0)
f1lo, f1hi = boot_f1(y_test, y_pred)

def sig4(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return np.nan
    return float(f"{x:.4g}")

metrics = pd.DataFrame([
    {"metric": primary, "value": sig4(value), "ci_low": sig4(lo), "ci_high": sig4(hi)},
    {"metric": "macro_f1", "value": sig4(f1v), "ci_low": sig4(f1lo), "ci_high": sig4(f1hi)},
])
mlca_runtime.save_table(metrics, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")
mlca_runtime.record_metric(primary, float(value), "test", ci=(float(lo), float(hi)))
if primary in ("roc_auc", "roc_auc_ovr"):
    mlca_runtime.record_fact_candidate("test_roc_auc", float(value), source="test_metrics.csv")

pred_df = pd.DataFrame({"row_id": test.index, "partition": "test", "y_true": y_test, "y_pred": y_pred})
for j, c in enumerate(classes):
    pred_df[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred_df, "test_predictions.parquet")

left_out = [c for c in classes if not np.any(y_test == c)]
print(f"train={train.shape} val={val.shape} test={test.shape}")
print(f"selected={selected} val_{primary}={best:.4f}")
print(f"test {primary}={value:.4f} CI=({lo:.4f},{hi:.4f})")
if left_out:
    print(f"test missing model classes: {left_out}")
print(f"saved rows={len(pred_df)}")
