import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.base import clone

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for key in ("task", "targets", "excluded_columns"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available: {list(protocol)}")
task, targets, excluded = protocol["task"], protocol["targets"], protocol["excluded_columns"]
group_col = protocol.get("group_column")
metric_name = task.get("primary_metric") or "silhouette"

train = pd.read_parquet(inp["train"])
with open(inp["05_preprocess_fit_preprocessing_MATQH9FA/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_retain_features_5RGESHAY/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

pre_cols = list(getattr(pre, "feature_names_in_", []))
sel_cols = list(getattr(sel, "feature_names_in_", []))

def make_frontend():
    if isinstance(sel, Pipeline):
        names = list(sel.named_steps)
        if any(n in ("preprocessor", "preprocess", "pre") for n in names) or (
            pre_cols and sel_cols and list(sel_cols) == list(pre_cols)
        ):
            return clone(sel)
        if pre_cols and sel_cols and list(sel_cols) != list(pre_cols):
            return Pipeline([("preprocessor", clone(pre)), ("selector", clone(sel))])
        return clone(sel)
    return Pipeline([("preprocessor", clone(pre)), ("selector", clone(sel))])

front = make_frontend()
if sel_cols and pre_cols and list(sel_cols) == list(pre_cols):
    feat_cols = sel_cols
elif pre_cols:
    feat_cols = pre_cols
elif sel_cols:
    feat_cols = sel_cols
else:
    drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
    feat_cols = []
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        if (s.dtype == object or str(s.dtype).startswith("string")) and s.nunique(dropna=True) == len(s):
            continue
        if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
            continue
        feat_cols.append(c)

allowed = set(train.columns) - set(targets) - set(excluded) - ({group_col} if group_col else set())
feat_cols = [c for c in feat_cols if c in allowed]
X = train[feat_cols].copy()
n = len(X)
rng = np.random.default_rng(42)

results = []
for k in range(2, 11):
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=1024)
    pipe = clone(front)
    pipe.steps.append(("kmeans", km))
    mlca_runtime.fit_record(pipe, X, partition="train")
    Xt = pipe[:-1].transform(X)
    labels = pipe.named_steps["kmeans"].predict(Xt)
    sil = silhouette_score(Xt, labels, sample_size=min(10000, n), random_state=42)
    aris = []
    for _ in range(5):
        w = rng.poisson(1, size=n).astype(float)
        if w.sum() == 0:
            w[0] = 1.0
        pb = clone(pipe)
        mlca_runtime.fit_record(pb, X, partition="train", kmeans__sample_weight=w)
        lb = pb.named_steps["kmeans"].predict(pb[:-1].transform(X))
        aris.append(adjusted_rand_score(labels, lb))
    results.append((k, float(sil), float(np.mean(aris)), pipe))

best = max(results, key=lambda r: (r[1], r[2]))
mlca_runtime.record_metric("selected_k", float(best[0]), "train")
mlca_runtime.record_metric(metric_name, best[1], "train")
mlca_runtime.record_metric("stability_ari", best[2], "train")
mlca_runtime.save_model(best[3], "alternative_model.pkl")
print("n", n, "k", best[0], metric_name, round(best[1], 4), "stability", round(best[2], 4))
