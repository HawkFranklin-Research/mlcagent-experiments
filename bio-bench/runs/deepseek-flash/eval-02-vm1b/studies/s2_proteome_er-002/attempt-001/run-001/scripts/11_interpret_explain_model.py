import json, pickle, warnings
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
if isinstance(targets, str):
    targets = [targets]
excluded_raw = protocol["excluded_columns"]
excluded = set(excluded_raw) if isinstance(excluded_raw, list) else set(excluded_raw.get("examples", []))
group = protocol.get("group_column")
task = protocol["task"]
n_classes = int(task.get("n_classes", 2))

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
print(f"train={train.shape} val={val.shape}")

def keylike(s):
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)):
        return False
    if s.nunique(dropna=False) != len(s):
        return False
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or any(k in s.name.lower() for k in ("id", "key", "sample", "subject", "code"))
    )

raw_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group and not keylike(train[c])
]
print(f"raw_features={len(raw_cols)}")

def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

cands = []
for key in ["latest_model", "07_model_train_logistic_RQ5H03SW/model.pkl"]:
    try:
        cands.append(load(inputs[key]))
    except Exception:
        pass

def feat_in(m):
    f = getattr(m, "feature_names_in_", None)
    if f is None and isinstance(m, Pipeline):
        f = getattr(m.steps[0][1], "feature_names_in_", None)
    return list(f) if f is not None else None

def classes_of(m):
    c = getattr(m, "classes_", None)
    if c is None and isinstance(m, Pipeline):
        c = getattr(m.steps[-1][1], "classes_", None)
    return c

def proba(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)
    d = np.asarray(m.decision_function(X))
    if d.ndim == 1:
        return 1.0 / (1.0 + np.exp(-d))
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def score(m, X, y):
    p = proba(m, X)
    c = classes_of(m)
    if p.ndim == 2 and p.shape[1] == 2:
        if c is not None and len(c) == 2:
            yb = (y == c[1]).astype(int)
        else:
            u = np.unique(y)
            yb = (y == u[-1]).astype(int)
        return roc_auc_score(yb, p[:, 1])
    if n_classes > 2:
        return roc_auc_score(y, p, multi_class="ovr", average="macro")
    if c is not None and len(c) == 2:
        yb = (y == c[1]).astype(int)
    else:
        yb = y
    return roc_auc_score(yb, p.ravel())

target = targets[0]
y = val[target].values
model = None
mf = None
for m in cands:
    f = feat_in(m) or list(raw_cols)
    if not all(c in val.columns for c in f):
        continue
    try:
        score(m, val[list(f)], y)
        model, mf = m, list(f)
        break
    except Exception:
        pass
if model is None:
    raise RuntimeError("No frozen model could score validation rows")

Xval = val[mf].copy()

def importance_rank(m):
    est = m.steps[-1][1] if isinstance(m, Pipeline) else m
    if hasattr(est, "coef_"):
        arr = np.asarray(est.coef_)
        imp = np.abs(arr).mean(axis=0) if arr.ndim > 1 else np.abs(arr)
    elif hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_)
    else:
        return None
    names = None
    if isinstance(m, Pipeline):
        try:
            names = m[:-1].get_feature_names_out(input_features=mf)
        except Exception:
            try:
                names = m[:-1].get_feature_names_out()
            except Exception:
                names = None
    if names is None:
        names = getattr(est, "feature_names_in_", None)
    if names is None and len(imp) == len(mf):
        names = mf
    if names is None or len(names) != len(imp):
        return None
    return [(str(n), float(v)) for n, v in zip(names, imp)]

rank = importance_rank(model)
top = []
if rank is not None:
    seen = set()
    for n, _ in sorted(rank, key=lambda x: -x[1]):
        r = n if n in mf else n.split("__")[-1]
        if r in mf and r not in seen:
            top.append(r)
            seen.add(r)
        if len(top) >= 50:
            break

if len(mf) <= 500:
    features, n_repeats = mf, 20
    method = "validation permutation importance full width"
elif top:
    features, n_repeats = top, 5
    method = "validation permutation importance top-50 by model importance"
else:
    features, n_repeats = [], 0
    method = "limitation: no reliable importance ranking"

base = score(model, Xval, y) if features else np.nan
rows = []
if features:
    rng = np.random.default_rng(42)
    for feat in features:
        vals = []
        for _ in range(n_repeats):
            Xp = Xval.copy()
            Xp[feat] = rng.permutation(Xp[feat].to_numpy())
            vals.append(base - score(model, Xp, y))
        rows.append((feat, float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0))

out = pd.DataFrame(rows, columns=["feature", "importance_mean", "importance_sd"])
out = out.sort_values("importance_mean", ascending=False).head(15).round(4)
mlca_runtime.save_table(out, "interpretation.csv", caption=method)
if len(out):
    topf = str(out.iloc[0]["feature"])
    mlca_runtime.record_fact_candidate("top_feature", topf, source="interpretation.csv")
    print(f"method={method} features={len(features)} repeats={n_repeats} top={topf}")
else:
    mlca_runtime.record_fact_candidate("top_feature", "unavailable", source="interpretation.csv")
    print(f"method={method} features=0 repeats=0 top=unavailable")
