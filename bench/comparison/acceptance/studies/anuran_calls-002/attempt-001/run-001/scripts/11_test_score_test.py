import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
SEED = 42

inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
task = protocol["task"]
primary = task["primary_metric"]
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(inp["train"])
test = pd.read_parquet(inp["test"])

LOG = "08_model_train_logistic_GP5KS1CK/model.pkl"
ALT = "09_model_train_alternative_16WYVMV8/alternative_model.pkl"
VAL = "10_validate_score_validation_XTD87YEX/validation_metrics.csv"

def unwrap(o):
    if isinstance(o, dict):
        for k in ("model", "pipeline", "estimator", "best_model"):
            if k in o:
                return unwrap(o[k])
    return o

def path_for(nm):
    s = str(nm).lower()
    if any(x in s for x in ("alt", "09", "gradient", "hist", "boost", "forest")):
        return inp[ALT]
    if any(x in s for x in ("log", "08")):
        return inp[LOG]
    return None

def pick_model():
    higher = primary not in {"brier_score", "log_loss", "mae", "mse", "rmse"}
    try:
        df = pd.read_csv(inp[VAL])
    except Exception:
        return inp["latest_model"]
    mcol = next((c for c in df.columns if str(c).lower() in
                 ("model", "candidate", "model_name", "estimator", "name")), None)
    for mc in df.columns:
        if str(mc).lower() == "metric":
            vc = next((c for c in df.columns if str(c).lower() in ("value", "score")), None)
            if vc is not None and mcol is not None:
                sub = df[df[mc].astype(str) == primary]
                sc = {}
                for _, r in sub.iterrows():
                    p = path_for(r[mcol])
                    if p is not None:
                        sc[p] = float(r[vc])
                if sc:
                    return max(sc, key=sc.get) if higher else min(sc, key=sc.get)
    if primary in df.columns and mcol is not None:
        sub = df[[mcol, primary]].dropna().sort_values(primary, ascending=not higher)
        for nm in sub[mcol]:
            p = path_for(nm)
            if p is not None:
                return p
    return inp["latest_model"]

def get_classes(est):
    if hasattr(est, "classes_"):
        return np.asarray(est.classes_)
    if hasattr(est, "named_steps"):
        last = list(est.named_steps.values())[-1]
        if hasattr(last, "classes_"):
            return np.asarray(last.classes_)
    raise AttributeError("Model has no classes_")

def get_features(est, frame):
    names = getattr(est, "feature_names_in_", None)
    if names is None and hasattr(est, "named_steps"):
        for st in est.named_steps.values():
            names = getattr(st, "feature_names_in_", None)
            if names is not None:
                break
    bad = set(targets) | set(excluded) | ({group} if group else set())
    if names is not None:
        if any(c in bad for c in names):
            raise ValueError("Saved model expects a target/excluded/group column")
        return list(names)
    n = len(frame)
    keep = []
    for c in frame.columns:
        if c in bad:
            continue
        s = frame[c]
        if (s.dtype == object or pd.api.types.is_string_dtype(s.dtype)
                or isinstance(s.dtype, pd.CategoricalDtype)) and s.nunique(dropna=True) == n:
            continue
        if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=True) == n:
            continue
        keep.append(c)
    return keep

model_path = pick_model()
with open(model_path, "rb") as f:
    model = unwrap(pickle.load(f))

features = get_features(model, train)
X_test = test.loc[:, features]
y_test = test[target].to_numpy()
classes = get_classes(model)
proba = model.predict_proba(X_test)
if proba.shape[1] != len(classes):
    raise ValueError("Probability columns do not align with model classes")

def macro_auc(y_true, proba, classes):
    y_true = np.asarray(y_true)
    per = []
    for i, c in enumerate(classes):
        yb = (y_true == c).astype(int)
        if yb.min() == yb.max():
            continue
        per.append(roc_auc_score(yb, proba[:, i]))
    return float(np.mean(per)) if per else float("nan")

auc = macro_auc(y_test, proba, classes)
pred = classes[np.argmax(proba, axis=1)]
f1 = float(f1_score(y_test, pred, labels=classes, average="macro", zero_division=0))

rng = np.random.default_rng(SEED)
n = len(y_test)
auc_boot, f1_boot = [], []
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yy, pp = y_test[idx], proba[idx]
    a = macro_auc(yy, pp, classes)
    if np.isfinite(a):
        auc_boot.append(a)
    f1_boot.append(f1_score(yy, classes[np.argmax(pp, axis=1)], labels=classes,
                            average="macro", zero_division=0))

def bound(vals):
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

auc_lo, auc_hi = bound(auc_boot)
f1_lo, f1_hi = bound(f1_boot)

out = pd.DataFrame([
    {"metric": primary, "value": auc, "ci_low": auc_lo, "ci_high": auc_hi},
    {"metric": "macro_f1", "value": f1, "ci_low": f1_lo, "ci_high": f1_hi},
])
mlca_runtime.save_table(out.round(4), "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")
ci_pass = (auc_lo, auc_hi) if np.isfinite(auc_lo) and np.isfinite(auc_hi) else None
mlca_runtime.record_metric(primary, float(auc), split="test", ci=ci_pass)
mlca_runtime.record_fact_candidate("test_roc_auc", float(auc), source="test_metrics.csv")

print(f"model={model_path.name} n_test={len(test)}")
print(f"{primary}={auc:.4f} CI=[{auc_lo:.4f},{auc_hi:.4f}] f1={f1:.4f}")
