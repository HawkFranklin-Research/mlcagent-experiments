import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing protocol key {k!r}; available top-level keys: {list(d)}")
    return d[k]

task = need(proto, "task")
primary = need(task, "primary_metric")
targets = need(proto, "targets")
excluded = need(proto, "excluded_columns")
group = proto.get("group_column")

train_df = pd.read_parquet(inputs["train"])
val_df = pd.read_parquet(inputs["val"])
test_df = pd.read_parquet(inputs["test"])

def pick_features(df):
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    keep = []
    for c in [c for c in df.columns if c not in drop]:
        s = df[c]
        is_text = s.dtype == object or str(s.dtype).startswith("string") or isinstance(s.dtype, pd.CategoricalDtype)
        if is_text and s.nunique(dropna=True) == len(df):
            continue
        if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=True) == len(df):
            if any(k in c.lower() for k in ("id", "key", "code", "row", "index")):
                continue
        keep.append(c)
    return keep

feat_cols = pick_features(train_df)

def load_pkl(key):
    return pickle.loads(inputs[key].read_bytes())

pre = load_pkl("05_preprocess_fit_preprocessing_Y1MSMN58/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_MBXQM04A/selector_all.pkl")
base = load_pkl("07_model_train_baseline_9FVS2S2P/model.pkl")
alt = load_pkl("08_model_train_alternative_D2KDW4KS/alternative_model.pkl")

def frontend():
    if isinstance(sel, Pipeline):
        return sel
    if isinstance(pre, Pipeline):
        if sel is None:
            return pre
        if any("select" in n.lower() or "feature" in n.lower() for n in pre.named_steps):
            return pre
        return Pipeline([("preprocessor", pre), ("selector", sel)])
    if pre is not None and sel is not None:
        return Pipeline([("preprocessor", pre), ("selector", sel)])
    return pre if pre is not None else sel

def complete(model):
    if isinstance(model, Pipeline):
        return model
    fe = frontend()
    steps = []
    if fe is not None:
        steps.append(("frontend", fe))
    steps.append(("model", model))
    return Pipeline(steps)

def transform(pipe, X):
    if isinstance(pipe, Pipeline) and len(pipe.steps) > 1:
        return Pipeline(pipe.steps[:-1]).transform(X)
    fe = frontend()
    return fe.transform(X) if fe is not None else X

def score(pipe, df):
    X = df[feat_cols]
    lab = np.asarray(pipe.predict(X))
    Xt = transform(pipe, X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    n = len(Xt)
    k = len(np.unique(lab))
    if n < 2 or k < 2:
        return lab, {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    sil = silhouette_score(Xt, lab, sample_size=min(10000, n), random_state=42)
    return lab, {
        "silhouette": float(sil),
        "calinski_harabasz": float(calinski_harabasz_score(Xt, lab)),
        "davies_bouldin": float(davies_bouldin_score(Xt, lab)),
    }

def primary_key(m):
    ml = m.lower().replace("-", "_")
    if "silhouette" in ml:
        return "silhouette"
    if "calinski" in ml:
        return "calinski_harabasz"
    if "davies" in ml:
        return "davies_bouldin"
    return "silhouette"

pk = primary_key(primary)
pipes = {"baseline": complete(base), "alternative": complete(alt)}
best = None
try:
    vm = pd.read_csv(inputs["09_validate_score_validation_R4H7PF4S/validation_metrics.csv"])
    if {"model", "metric", "value"} <= set(vm.columns):
        m = vm["metric"].astype(str).str.lower().str.replace("-", "_")
        rows = vm[m == primary.lower().replace("-", "_")].copy()
        def cand_name(s):
            s = str(s).lower()
            if "alternative" in s or "08_" in s:
                return "alternative"
            if "baseline" in s or "07_" in s:
                return "baseline"
            return None
        if len(rows):
            rows["cand"] = rows["model"].map(cand_name)
            rows = rows.dropna(subset=["cand"])
            if len(rows):
                best = rows.sort_values("value", ascending=(pk == "davies_bouldin")).iloc[0]["cand"]
except Exception:
    best = None

if best is None:
    val_scores = {name: score(pipe, val_df)[1][pk] for name, pipe in pipes.items()}
    def sel_key(name):
        v = val_scores[name]
        if pd.isna(v):
            return -np.inf
        return -v if pk == "davies_bouldin" else v
    best = max(val_scores, key=sel_key)

pipe = pipes[best]
test_lab, test_met = score(pipe, test_df)
train_lab = np.asarray(pipe.predict(train_df[feat_cols]))

order = [pk] + [m for m in ("silhouette", "calinski_harabasz", "davies_bouldin") if m != pk]
rows = []
for m in order:
    v = test_met[m]
    rows.append({"model": best, "metric": m, "value": np.nan if pd.isna(v) else float(f"{v:.4g}")})
test_metrics = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(test_metrics, "test_metrics.csv")

assignments = pd.concat([
    pd.DataFrame({"row_id": train_df.index, "partition": "train", "cluster": train_lab}),
    pd.DataFrame({"row_id": test_df.index, "partition": "test", "cluster": test_lab}),
], ignore_index=True)
mlca_runtime.save_row_outputs(assignments, "assignments.parquet")

mlca_runtime.record_metric(primary, float(test_met[pk]), split="test")
mlca_runtime.record_fact_candidate("test_score", float(test_met[pk]), source="test_metrics.csv")
print("features", len(feat_cols), "train", len(train_df), "test", len(test_df))
print("selected", best)
print("test", primary, test_met[pk])
