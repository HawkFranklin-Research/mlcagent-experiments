import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, roc_curve, f1_score,
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing key {k!r}; available keys: {sorted(d.keys())}")
    return d[k]

protocol = json.loads(need(inp, "_protocol").read_text())
task = need(protocol, "task")
targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group = protocol.get("group_column", None)
primary_metric = need(task, "primary_metric")
target = targets[0]

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
val = val.loc[val[target].notna()].copy()

def feature_columns(df):
    skip = set(targets) | set(excluded)
    if group is not None:
        skip.add(group)
    cols, n = [], len(df)
    for c in df.columns:
        if c in skip:
            continue
        s = df[c]
        if s.nunique(dropna=False) == n and not pd.api.types.is_numeric_dtype(s):
            continue
        if s.nunique(dropna=False) == n and pd.api.types.is_integer_dtype(s):
            v = s.dropna().sort_values().to_numpy()
            if len(v) > 1 and np.all(np.diff(v) == 1):
                continue
        cols.append(c)
    return cols

feat = feature_columns(train)

pre = load_pkl(inp["05_preprocess_fit_preprocessing_WBRZJAT5/preprocessor.pkl"])
sel = load_pkl(inp["06_features_select_features_PDWSHJGW/selector.pkl"])
m1 = load_pkl(inp["07_model_train_logistic_H2ZAD1XK/model.pkl"])
m2 = load_pkl(inp["08_model_train_alternative_XPWMPVN8/alternative_model.pkl"])

def _has_pre(o):
    return isinstance(o, Pipeline) and any(("preprocess" in n or "preproc" in n) for n in o.named_steps)

def assemble(m, pre, sel):
    if isinstance(m, Pipeline):
        names = list(m.named_steps)
        hp = any("preprocess" in n or "preproc" in n for n in names)
        hs = any("select" in n for n in names)
        if hp and hs:
            return m
        if hp and not hs:
            if _has_pre(sel):
                return m
            steps = list(m.steps)
            steps.insert(-1, ("selector", sel))
            return Pipeline(steps)
        return m
    if _has_pre(sel):
        return Pipeline([("selector", sel), ("model", m)])
    return Pipeline([("preprocessor", pre), ("selector", sel), ("model", m)])

candidates = [("logistic", assemble(m1, pre, sel)), ("alternative", assemble(m2, pre, sel))]

def input_columns(pipe):
    try:
        names = list(pipe.feature_names_in_)
        if names:
            return names
    except Exception:
        pass
    return feat

def est_of(pipe):
    return pipe.steps[-1][1] if isinstance(pipe, Pipeline) else pipe

def align_y(model, y):
    classes = list(model.classes_)
    y_u = set(y.unique())
    if y_u <= set(classes):
        return y, classes
    if len(y_u) == len(classes):
        mapping = dict(zip(sorted(y_u), sorted(classes)))
        return y.map(mapping), classes
    raise ValueError(f"Cannot align y {sorted(y_u)} to classes {classes}")

rows = []
for name, pipe in candidates:
    X = val[input_columns(pipe)]
    est = est_of(pipe)
    y, classes = align_y(est, val[target])
    proba = np.asarray(pipe.predict_proba(X), dtype=float)
    if len(classes) == 2:
        pos = classes[1]
        idx = classes.index(pos)
        p = proba[:, idx]
        yb = (y == pos).astype(int).to_numpy()
        if len(np.unique(yb)) < 2:
            roc = pr = brier = bal = th = np.nan
        else:
            roc = roc_auc_score(yb, p)
            pr = average_precision_score(yb, p)
            brier = brier_score_loss(yb, p)
            fpr, tpr, thr = roc_curve(yb, p)
            bi = int(np.argmax(tpr - fpr))
            th = float(thr[bi])
            if not np.isfinite(th):
                th = float(np.nextafter(np.max(p), np.inf))
            bal = balanced_accuracy_score(yb, (p >= th).astype(int))
        rows.extend([
            (name, "roc_auc", roc, np.nan),
            (name, "pr_auc", pr, np.nan),
            (name, "balanced_accuracy", bal, th),
            (name, "brier_score", brier, np.nan),
        ])
    else:
        present = [c for c in classes if c in set(y.unique())]
        left = [c for c in classes if c not in set(y.unique())]
        if left:
            print(f"{name} missing classes: {left}")
        aucs = []
        for c in present:
            yc = (y == c).astype(int).to_numpy()
            if len(np.unique(yc)) == 2:
                aucs.append(roc_auc_score(yc, proba[:, classes.index(c)]))
        roc = float(np.mean(aucs)) if aucs else np.nan
        f1 = f1_score(y, pipe.predict(X), labels=classes, average="macro", zero_division=0)
        rows.extend([(name, "roc_auc_ovr", roc, np.nan), (name, "f1_macro", f1, np.nan)])

df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
for c in ["value", "threshold"]:
    df[c] = df[c].map(lambda v: np.nan if pd.isna(v) else float(f"{v:.4g}"))
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics per model")

prim = [r for r in rows if r[1] == primary_metric and np.isfinite(r[2])]
if not prim:
    raise ValueError(f"No finite {primary_metric!r}; metrics={sorted(set(r[1] for r in rows))}")
direction = -1 if "brier" in primary_metric.lower() else 1
prim_sorted = sorted(prim, key=lambda r: r[2], reverse=(direction == -1))
for r in prim:
    mlca_runtime.record_metric(primary_metric, float(r[2]), "val")
best = prim_sorted[-1]
roc_row = next(r for r in rows if r[0] == best[0] and r[1] == "roc_auc" and np.isfinite(r[2]))
mlca_runtime.record_fact_candidate("best_model", best[0], source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(roc_row[2]), source="validation_metrics.csv")

print(f"val rows {len(val)}; features {len(feat)}; models {len(candidates)}")
print(f"best {best[0]} {primary_metric}={best[2]:.4f}")
