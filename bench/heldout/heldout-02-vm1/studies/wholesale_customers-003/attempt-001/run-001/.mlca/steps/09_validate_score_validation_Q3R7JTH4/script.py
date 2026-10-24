import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())
for k in ("task", "targets", "excluded_columns"):
    if k not in proto:
        raise KeyError(f"Missing protocol key {k!r}; available: {list(proto)}")
primary = proto["task"]["primary_metric"]
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group = proto.get("group_column")
train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

def load_pkl(key):
    with open(paths[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("05_preprocess_fit_preprocessing_HBN850GP/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_5BEJHTH5/selector_all.pkl")
models = [
    ("baseline", load_pkl("07_model_train_baseline_V5KHW25A/model.pkl"),
     "07_model_train_baseline_V5KHW25A/manifest.json"),
    ("alternative", load_pkl("08_model_train_alternative_78714TF3/alternative_model.pkl"),
     "08_model_train_alternative_78714TF3/manifest.json"),
]

drop = set(targets) | set(excluded) | ({group} if group else set())
def key_like(s, name):
    if s.nunique(dropna=False) != len(s) or pd.api.types.is_float_dtype(s):
        return False
    lname = str(name).lower()
    if any(t in lname for t in ("id", "key", "index", "row")):
        return True
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        return bool(s.is_monotonic_increasing)
    return False

feature_cols = [c for c in train.columns if c not in drop and not key_like(train[c], c)]
feature_cols = [c for c in feature_cols if c in val.columns]
if not feature_cols:
    raise ValueError("No feature columns remain")

def names(obj):
    return list(getattr(obj, "feature_names_in_", []))

def pick(df, obj):
    ns = names(obj)
    if ns and all(n in df.columns for n in ns):
        return df[ns]
    return df[feature_cols]

def transformed(df):
    base = df[feature_cols]
    if sel is not None:
        sns = names(sel)
        if sns and all(n in df.columns for n in sns):
            return sel.transform(df[sns])
        if pre is not None:
            return sel.transform(pre.transform(pick(df, pre)))
        return sel.transform(base)
    if pre is not None:
        return pre.transform(pick(df, pre))
    return base

def labels_space(model, df):
    if isinstance(model, Pipeline):
        Xin = pick(df, model)
        try:
            lab = model.predict(Xin)
            try:
                Xm = model[:-1].transform(Xin)
            except Exception:
                Xm = Xin
            return np.asarray(lab), Xm
        except Exception:
            pass
    ns = names(model)
    if ns and all(n in df.columns for n in ns):
        Xin = df[ns]
        return np.asarray(model.predict(Xin)), Xin
    Xt = transformed(df)
    return np.asarray(model.predict(Xt)), Xt

def r4(x):
    return float(f"{x:.4g}") if np.isfinite(x) else np.nan

def find_stability(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "stability" in str(k).lower() and isinstance(v, (int, float)):
                return float(v)
        for v in obj.values():
            r = find_stability(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_stability(v)
            if r is not None:
                return r
    return None

rows, vals = [], {}
for name, model, man_key in models:
    lab, Xm = labels_space(model, val)
    n, k = len(lab), len(np.unique(lab))
    if n >= 2 and 1 < k < n:
        sil = silhouette_score(Xm, lab, sample_size=min(10000, n), random_state=42)
        ch = calinski_harabasz_score(Xm, lab)
        db = davies_bouldin_score(Xm, lab)
    else:
        sil = ch = db = np.nan
    d = {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}
    vals[name] = d
    for metric in ("silhouette", "calinski_harabasz", "davies_bouldin"):
        rows.append({"model": name, "metric": metric, "value": r4(d[metric])})
    st = find_stability(json.loads(paths[man_key].read_text()))
    if st is not None:
        rows.append({"model": name, "metric": "stability", "value": r4(st)})

direction = {"silhouette": "max", "calinski_harabasz": "max", "davies_bouldin": "min"}
if primary not in direction:
    raise ValueError(f"Unsupported primary_metric {primary!r}; known: {list(direction)}")
if direction[primary] == "max":
    best_name = max(vals, key=lambda n: vals[n][primary] if np.isfinite(vals[n][primary]) else -np.inf)
else:
    best_name = min(vals, key=lambda n: vals[n][primary] if np.isfinite(vals[n][primary]) else np.inf)
best_score = float(vals[best_name][primary])

df = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation clustering metrics")
mlca_runtime.record_metric(primary, best_score, "val")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
print(f"val rows={len(val)} features={len(feature_cols)}")
print(f"candidates={len(vals)}")
print(f"best={best_name} {primary}={best_score:.4g}")
