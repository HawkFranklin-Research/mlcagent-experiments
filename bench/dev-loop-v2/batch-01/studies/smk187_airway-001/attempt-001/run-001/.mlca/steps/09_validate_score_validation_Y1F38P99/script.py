import mlca_runtime as rt
import json, pickle, numpy as np, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, roc_curve

rt.set_seed(42)
inp = rt.inputs()
protocol = json.loads(Path(inp["_protocol"]).read_text())
for k in ["targets", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing protocol key {k!r}; available keys: {sorted(protocol)}")
targets = list(protocol["targets"])
task = protocol["task"]
primary = task["primary_metric"]
task_type = task.get("type")
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column", None)

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
drop = set(targets + excluded + ([group] if group else []))
feature_cols = [c for c in train.columns if c not in drop]
for c in list(feature_cols):
    s = train[c]
    if (s.dtype == object or str(s.dtype).startswith("string")) and s.nunique(dropna=True) == len(s):
        feature_cols.remove(c)

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inp["05_preprocess_fit_preprocessing_TZWQXPN7/preprocessor.pkl"])
sel = load_pkl(inp["06_features_select_features_NTK41RPZ/selector.pkl"])
cand = [("logistic", inp["07_model_train_logistic_0MNK5WE5/model.pkl"]),
        ("alternative", inp["08_model_train_alternative_4KBDBVMX/alternative_model.pkl"])]
lm = inp.get("latest_model")
if lm and Path(lm).resolve() not in {Path(p).resolve() for _, p in cand}:
    cand.append(("latest_model", lm))
models = [(n, load_pkl(p)) for n, p in cand]

def input_cols(obj):
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        try:
            return list(obj.feature_names_in_)
        except Exception:
            pass
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
            break
    return None

def transform(X, obj, fallback):
    cols = input_cols(obj) or fallback
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"Missing columns for {type(obj).__name__}: {missing[:3]}")
    out = obj.transform(X.loc[:, cols])
    if isinstance(out, pd.DataFrame):
        return out
    try:
        names = obj.get_feature_names_out()
        if len(names) == out.shape[1]:
            return pd.DataFrame(out, columns=names, index=X.index)
    except Exception:
        pass
    return out

def proba(model, X, cols, pos_label, index):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=cols, index=index)
    elif cols is not None:
        X = X.loc[:, cols]
    pr = model.predict_proba(X)
    cls = list(model.classes_)
    if pos_label not in cls:
        raise ValueError(f"Positive label {pos_label!r} not in model classes {cls}")
    return pr[:, cls.index(pos_label)]

first = models[0][1]
pos_label = task.get("positive_label", None)
if pos_label is None:
    pos_label = list(first.classes_)[1]
y = val[targets[0]]
y_bin = (y == pos_label).astype(int)

def scorer_for(model):
    mcols = input_cols(model)
    raw_set = set(feature_cols)
    if mcols is not None and set(mcols).issubset(raw_set):
        return lambda X: proba(model, X, mcols, pos_label, X.index)
    sel_cols = input_cols(sel)
    if sel is not None and sel_cols is not None and set(sel_cols).issubset(raw_set):
        return lambda X: proba(model, transform(X, sel, feature_cols), mcols, pos_label, X.index)
    if pre is not None and sel is not None:
        def f(X):
            Xp = transform(X, pre, feature_cols)
            fb = list(Xp.columns) if isinstance(Xp, pd.DataFrame) else feature_cols
            return proba(model, transform(Xp, sel, fb), mcols, pos_label, X.index)
        return f
    if pre is not None:
        return lambda X: proba(model, transform(X, pre, feature_cols), mcols, pos_label, X.index)
    return lambda X: proba(model, X, mcols or feature_cols, pos_label, X.index)

def rs(x):
    return float(f"{x:.4g}") if np.isfinite(x) else np.nan

rows, val_primary = [], {}
for name, model in models:
    p = scorer_for(model)(val)
    auc = roc_auc_score(y_bin, p)
    ap = average_precision_score(y_bin, p)
    fpr, tpr, thr = roc_curve(y_bin, p)
    i = int(np.argmax(tpr - fpr))
    t = float(thr[i])
    if not np.isfinite(t):
        t = float(np.nextafter(p.max(), np.inf)) if len(p) else 0.0
    neg_label = [c for c in list(model.classes_) if c != pos_label][0]
    pred = np.where(p >= t, pos_label, neg_label)
    bal = balanced_accuracy_score(y, pred)
    brier = brier_score_loss(y_bin, p)
    for metric, value, th in [("roc_auc", auc, np.nan), ("pr_auc", ap, np.nan),
                              ("balanced_accuracy", bal, rs(t)), ("brier_score", brier, np.nan)]:
        rows.append({"model": name, "metric": metric, "value": rs(value), "threshold": th})
    val_primary[name] = auc

higher = {"roc_auc": True, "pr_auc": True, "balanced_accuracy": True, "brier_score": False}.get(primary, True)
best_name = max(val_primary, key=val_primary.get) if higher else min(val_primary, key=val_primary.get)
best_val = val_primary[best_name]
df = pd.DataFrame(rows)
rt.save_table(df, "validation_metrics.csv", caption="Validation metrics by candidate model")
rt.record_metric(primary, float(best_val), "val")
rt.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
rt.record_fact_candidate("best_val_roc_auc", float(best_val), source="validation_metrics.csv")
print(f"val rows={len(val)} features={len(feature_cols)} models={len(models)}")
print(f"best={best_name} {primary}={best_val:.4f}")
