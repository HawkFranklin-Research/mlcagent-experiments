import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
targets = proto.get("targets") or []
excluded = proto.get("excluded_columns") or []
group = proto.get("group_column")

train = pd.read_parquet(inp["train"])
feat = [c for c in train.columns if c not in targets and c not in excluded and c != group]
keep = []
for c in feat:
    s = train[c]
    if s.dtype.kind in "iu" and s.nunique() == len(s) and len(s) > 20:
        continue
    if s.dtype == object and s.nunique() == len(s):
        continue
    keep.append(c)
feat = keep
if not feat:
    raise ValueError("no features after exclusions")
X = train[feat].copy()

with inp["05_preprocess_fit_preprocessing_XP5A8XH9/preprocessor.pkl"].open("rb") as f:
    pre = pickle.load(f)
with inp["06_features_retain_features_MWAQCCQG/selector_all.pkl"].open("rb") as f:
    sel = pickle.load(f)

def has_ct(o):
    if isinstance(o, ColumnTransformer):
        return True
    if isinstance(o, Pipeline):
        return any(has_ct(s) for _, s in o.named_steps.items())
    return False

front = sel if has_ct(sel) else Pipeline([("pre", pre), ("sel", sel)])

rng = np.random.RandomState(42)
n = len(X)
res = []
for k in range(2, 7):
    pipe = Pipeline([("front", front), ("km", KMeans(n_clusters=k, n_init=10, random_state=42))])
    fitted = mlca_runtime.fit_record(pipe, X, "train")
    Xt = fitted[:-1].transform(X)
    lab = fitted[-1].labels_
    if len(np.unique(lab)) < 2:
        continue
    s = float(silhouette_score(Xt, lab, sample_size=None if n <= 10000 else 10000, random_state=42))
    aris = []
    for b in range(3):
        w = rng.multinomial(n, np.ones(n) / n).astype(float)
        p2 = Pipeline([("front", front), ("km", KMeans(n_clusters=k, n_init=5, random_state=100 + b))])
        f2 = mlca_runtime.fit_record(p2, X, "train", km__sample_weight=w)
        aris.append(float(adjusted_rand_score(lab, f2[-1].labels_)))
    ari = float(np.mean(aris))
    res.append({"k": k, "silhouette": s, "stability_ari": ari, "score": s * max(ari, 0.0)})

if not res:
    raise RuntimeError("no valid clusterings")
res.sort(key=lambda r: r["score"], reverse=True)
best = res[0]
best_k = int(best["k"])

final = Pipeline([("front", front), ("km", KMeans(n_clusters=best_k, n_init=10, random_state=42))])
mlca_runtime.fit_record(final, X, "train")
mlca_runtime.save_model(final, "model.pkl")
mlca_runtime.record_metric("silhouette", best["silhouette"], "train")
mlca_runtime.record_metric("stability_ari", best["stability_ari"], "train")
mlca_runtime.save_table(pd.DataFrame(res).round(4), "cluster_selection", "KMeans candidate comparison")
print("feat", len(feat), "k", best_k, "sil", round(best["silhouette"], 3))
