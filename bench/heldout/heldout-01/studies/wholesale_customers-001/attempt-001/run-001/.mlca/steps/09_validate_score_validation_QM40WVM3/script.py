import mlca_runtime, json, pickle, re
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
protocol = json.loads(inp["_protocol"].read_text())

if "task" not in protocol:
    raise KeyError(f"Missing task; available top-level keys: {sorted(protocol)}")
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing primary_metric; available top-level keys: {sorted(protocol)}")
primary = task["primary_metric"]
if not isinstance(primary, str):
    raise ValueError("task.primary_metric must be a string")

targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")

def key_like(s):
    low = s.name.lower()
    if re.search(r'(^|_)(row_?id|id|index|key|uuid|guid)$', low):
        return True
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or
            isinstance(s.dtype, pd.CategoricalDtype)):
        return s.nunique(dropna=True) == len(s)
    return False

drop = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in drop and not key_like(train[c])]
if not feature_cols:
    raise ValueError("No feature columns remain after exclusions")
missing = [c for c in feature_cols if c not in val.columns]
if missing:
    raise KeyError(f"Validation missing feature columns: {missing}")

def load_pre_sel():
    with inp["05_preprocess_fit_preprocessing_Q63KMMVN/preprocessor.pkl"].open("rb") as f:
        pre = pickle.load(f)
    with inp["06_features_retain_features_S31FQA6H/selector_all.pkl"].open("rb") as f:
        sel = pickle.load(f)
    return pre, sel

def model_inputs(model, df):
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        if all(c in df.columns for c in cols):
            return df[cols].copy(), True
    return df[feature_cols].copy(), False

def predict_labels(model, Xraw, raw_ok):
    if raw_ok:
        return np.asarray(model.predict(Xraw))
    pre, sel = load_pre_sel()
    Xt = pre.transform(Xraw) if hasattr(pre, "transform") else Xraw
    Xt = sel.transform(Xt) if hasattr(sel, "transform") else Xt
    return np.asarray(model.predict(Xt))

def cluster_features(model, Xraw, raw_ok):
    if raw_ok:
        if isinstance(model, Pipeline) and len(model.named_steps) > 1:
            try:
                return model[:-1].transform(Xraw)
            except Exception:
                pass
        return Xraw
    pre, sel = load_pre_sel()
    Xt = pre.transform(Xraw) if hasattr(pre, "transform") else Xraw
    Xt = sel.transform(Xt) if hasattr(sel, "transform") else Xt
    return Xt

def stability(man):
    vals, stack = [], [man]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, (dict, list)):
                    stack.append(v)
                elif isinstance(v, (int, float)) and "stability" in k.lower():
                    vals.append(float(v))
        elif isinstance(x, list):
            stack.extend(x)
    return max(vals) if vals else -np.inf

cand_defs = [
    ("baseline", "07_model_train_baseline_0H9X73N5/model.pkl", "07_model_train_baseline_0H9X73N5/manifest.json"),
    ("alternative", "08_model_train_alternative_6TXXHVGF/alternative_model.pkl", "08_model_train_alternative_6TXXHVGF/manifest.json"),
]
cands = []
for name, mkey, mankey in cand_defs:
    with inp[mkey].open("rb") as f:
        model = pickle.load(f)
    man = json.loads(inp[mankey].read_text())
    cands.append((name, model, stability(man)))

rows, by_model = [], {}
for name, model, st in cands:
    Xraw, raw_ok = model_inputs(model, val)
    labels = predict_labels(model, Xraw, raw_ok)
    Xt = cluster_features(model, Xraw, raw_ok)
    idx = (np.arange(len(labels)) if len(labels) <= 10000
           else np.random.RandomState(42).choice(len(labels), 10000, replace=False))
    ys = np.asarray(labels)[idx]
    if hasattr(Xt, "iloc"):
        Xs = Xt.iloc[idx].to_numpy()
    elif hasattr(Xt, "toarray") or hasattr(Xt, "tocsr"):
        Xs = Xt[idx]
        if hasattr(Xs, "toarray"):
            Xs = Xs.toarray()
    else:
        Xs = np.asarray(Xt)[idx]
    Xs = np.asarray(Xs)
    if Xs.ndim == 1:
        Xs = Xs.reshape(-1, 1)
    u = np.unique(ys)
    if len(u) < 2 or len(u) >= len(ys):
        sil = ch = db = np.nan
    else:
        try:
            sil = silhouette_score(Xs, ys, sample_size=min(10000, len(ys)), random_state=42)
            ch = calinski_harabasz_score(Xs, ys)
            db = davies_bouldin_score(Xs, ys)
        except Exception:
            sil = ch = db = np.nan
    vals = {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}
    by_model[name] = vals
    for m, v in vals.items():
        rows.append({"model": name, "metric": m,
                     "value": float(f"{v:.4g}") if np.isfinite(v) else np.nan})

df = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation clustering metrics")

primary_vals = []
for n, model, st in cands:
    d = by_model[n]
    if primary in d and np.isfinite(d[primary]):
        primary_vals.append((n, d[primary], st))
if not primary_vals:
    raise RuntimeError("No finite primary validation metric")

direction = -1 if "davies" in primary.lower() else 1
best = None
for n, v, st in primary_vals:
    key = (direction * v, st)
    if best is None or key > best[0]:
        best = (key, n, v)
_, best_model, best_val = best

mlca_runtime.record_metric(primary, float(best_val), split="val")
mlca_runtime.record_fact_candidate("best_validation_score", float(best_val), source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")

print(f"train={train.shape} val={val.shape} features={len(feature_cols)}")
print(f"candidates={len(cands)} metric_rows={len(df)}")
print(f"val_sample={min(10000, len(val))}")
print(f"best={best_model} {primary}={best_val:.6g}")
