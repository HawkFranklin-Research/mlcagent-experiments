import json, pickle, numpy as np, pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.base import clone

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {sorted(d)}")
    return d[k]

targets = set(req(proto, "targets"))
excluded = set(req(proto, "excluded_columns"))
group = proto.get("group_column")
_ = req(proto, "task")
train = pd.read_parquet(inp["train"])
with open(inp["05_preprocess_fit_preprocessing_WG30AQ6C/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_retain_features_SSZF8P1C/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def in_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "named_steps"):
        for s in obj.named_steps.values():
            if hasattr(s, "feature_names_in_"):
                return list(s.feature_names_in_)
    return None

pre_cols, sel_cols = in_cols(pre), in_cols(sel)
base, raw_cols = pre, pre_cols
if sel is not None and hasattr(sel, "named_steps") and sel_cols is not None and all(c in train.columns for c in sel_cols):
    base, raw_cols = sel, sel_cols
elif sel is not None and sel_cols is not None and not all(c in train.columns for c in sel_cols):
    try:
        out = list(pre.get_feature_names_out(raw_cols))
    except Exception:
        out = []
    if out and set(sel_cols).issubset(set(out)):
        base = Pipeline([("pre", pre), ("sel", sel)])

if raw_cols is None:
    banned = targets | excluded
    if group is not None:
        banned.add(group)
    raw_cols, n = [], len(train)
    for c in train.columns:
        if c in banned:
            continue
        s = train[c]
        if (s.dtype == object or str(s.dtype).startswith("category")) and s.nunique(dropna=False) >= n:
            continue
        if any(t in str(c).lower() for t in ("id", "key", "code")) and s.nunique(dropna=False) >= n:
            continue
        raw_cols.append(c)
else:
    raw_cols = [c for c in raw_cols if c in train.columns]

bad = [c for c in raw_cols if c in targets or c in excluded or (group is not None and c == group)]
if bad or not raw_cols:
    raise ValueError(f"Invalid feature columns: {bad}")

X, n = train[raw_cols], len(train)
rng = np.random.default_rng(42)
cands = []
for k in range(2, 9):
    pipe = Pipeline([
        ("front", clone(base)),
        ("scale", StandardScaler(with_mean=False)),
        ("km", KMeans(n_clusters=k, random_state=42, n_init=10)),
    ])
    ref = mlca_runtime.fit_record(pipe, X, "train")
    labels = ref.predict(X)
    Xt = ref[:-1].transform(X)
    idx = np.arange(n)
    if n > 100000:
        idx = rng.choice(n, size=10000, replace=False)
        print(f"metric subsample n={len(idx)}")
    Xm, lm = Xt[idx], labels[idx]
    sil = silhouette_score(Xm, lm, sample_size=min(10000, len(idx)), random_state=42)
    db = davies_bouldin_score(Xm, lm)
    ch = calinski_harabasz_score(Xm, lm)
    aris = []
    for _ in range(5):
        w = rng.multinomial(n, np.full(n, 1.0 / n))
        bpipe = mlca_runtime.fit_record(clone(pipe), X, "train", sample_weight=w)
        aris.append(adjusted_rand_score(labels, bpipe.predict(X)))
    cands.append((k, sil, db, ch, float(np.mean(aris)), ref))

pool = [c for c in cands if c[4] >= 0.5] or cands
k, sil, db, ch, stab, ref = max(pool, key=lambda c: (c[1], c[4], -c[2], c[3]))
mlca_runtime.save_model(ref, "model.pkl")
for name, value in [("n_clusters", k), ("silhouette", sil), ("davies_bouldin", db),
                    ("calinski_harabasz", ch), ("stability_ari", stab)]:
    mlca_runtime.record_metric(name, float(value), "train")
print(f"selected k={k} silhouette={sil:.4f} stability={stab:.4f}")
