import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
SEED = 42
inp = mlca_runtime.inputs()

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing {k!r}; available: {sorted(d.keys())}")
    return d[k]

protocol = json.loads(Path(inp["_protocol"]).read_text())
targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
group = req(protocol, "group_column")
task = req(protocol, "task")
primary = req(task, "primary_metric")
target = targets[0]

train = pd.read_parquet(inp["train"])
test = pd.read_parquet(inp["test"])

def feature_cols(df):
    skip = set(targets) | set(excluded) | ({group} if group else set())
    cols = []
    for c in df.columns:
        if c in skip:
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s) and s.is_unique:
            continue
        if s.is_unique and ("id" in str(c).lower() or "index" in str(c).lower()):
            continue
        cols.append(c)
    return cols

raw_cols = feature_cols(train)

def load_pkl(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("06_preprocess_fit_preprocessing_XZ1WZNJH/preprocessor.pkl")
sel = load_pkl("07_features_select_features_KRQ7615Z/selector.pkl")
if hasattr(pre, "feature_names_in_"):
    raw_cols = list(pre.feature_names_in_)

candidates = [
    ("logistic", "08_model_train_logistic_6YJ0NC7Y/model.pkl", "10_validate_score_validation_B4M4NX4R/validation_metrics.csv"),
    ("alternative", "09_model_train_alternative_143BSS6M/alternative_model.pkl", "11_validate_score_validation_9RDBMSFC/validation_metrics.csv"),
]

def read_metric(path, name):
    df = pd.read_csv(path)
    if "metric" in df.columns:
        m = df[df["metric"] == name]
        if len(m):
            return float(m["value"].iloc[0])
    if name in df.columns:
        return float(df[name].iloc[0])
    raise ValueError(f"Metric {name!r} not found in {path}")

lower_is_better = any(x in primary.lower() for x in ["brier", "loss", "error", "mae", "mse", "rmse"])
scored = []
for name, model_key, val_key in candidates:
    scored.append((read_metric(inp[val_key], primary), name, model_key))
selected = min(scored, key=lambda t: t[0]) if lower_is_better else max(scored, key=lambda t: t[0])
_, selected_name, model_key = selected
model = load_pkl(model_key)

def predict_proba(model, pre, sel, Xraw):
    if hasattr(model, "feature_names_in_"):
        mcols = list(model.feature_names_in_)
        if all(c in Xraw.columns for c in mcols):
            return model.predict_proba(Xraw[mcols])
    tried = []
    Xp = pre.transform(Xraw[raw_cols]) if pre is not None else Xraw[raw_cols]
    if sel is not None:
        tried.append(sel.transform(Xp))
    tried.append(Xp)
    if sel is not None:
        tried.append(sel.transform(Xraw[raw_cols]))
    for Xc in tried:
        if hasattr(model, "n_features_in_") and Xc.shape[1] != model.n_features_in_:
            continue
        try:
            p = model.predict_proba(Xc)
            if p.ndim == 2:
                return p
        except Exception:
            continue
    raise RuntimeError("Could not compute probabilities from saved artifacts")

proba = predict_proba(model, pre, sel, test[raw_cols].copy())
classes = np.asarray(model.classes_)
y_true = test[target].to_numpy()

def macro_ovr_auc(y, p, cls):
    present = [c for c in cls if np.any(y == c)]
    if len(present) == len(cls):
        return float(roc_auc_score(y, p, multi_class="ovr", average="macro", labels=cls))
    vals = []
    for c in present:
        j = int(np.where(cls == c)[0][0])
        vals.append(float(roc_auc_score(y == c, p[:, j])))
    return float(np.mean(vals)) if vals else np.nan

def macro_f1(y, p, cls):
    pred = cls[np.argmax(p, axis=1)]
    return float(f1_score(y, pred, average="macro", labels=cls, zero_division=0))

def boot_ci(fn, y, p, cls, n=1000):
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        v = fn(y[idx], p[idx], cls)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)

primary_val = macro_ovr_auc(y_true, proba, classes) if primary == "roc_auc_ovr" else float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes))
primary_lo, primary_hi = boot_ci(macro_ovr_auc, y_true, proba, classes)
f1_val = macro_f1(y_true, proba, classes)
f1_lo, f1_hi = boot_ci(macro_f1, y_true, proba, classes)

def sig4(x):
    return float(f"{x:.4g}") if np.isfinite(x) else np.nan

metrics = pd.DataFrame([
    {"metric": primary, "value": sig4(primary_val), "ci_low": sig4(primary_lo), "ci_high": sig4(primary_hi)},
    {"metric": "macro_f1", "value": sig4(f1_val), "ci_low": sig4(f1_lo), "ci_high": sig4(f1_hi)},
])
mlca_runtime.save_table(metrics, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")
mlca_runtime.record_metric(primary, primary_val, "test", ci=(primary_lo, primary_hi))
mlca_runtime.record_metric("macro_f1", f1_val, "test", ci=(f1_lo, f1_hi))
if primary in ("roc_auc", "roc_auc_ovr"):
    mlca_runtime.record_fact_candidate("test_roc_auc", sig4(primary_val), source="test_metrics.csv")
print(f"selected={selected_name} n_test={len(test)}")
print(f"{primary}={primary_val:.4g} CI=({primary_lo:.4g},{primary_hi:.4g})")
print(f"macro_f1={f1_val:.4g}")
