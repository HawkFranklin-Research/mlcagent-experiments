import mlca_runtime
import json, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from scipy.stats import rankdata

def need(d, k):
    if k not in d:
        raise KeyError(f"missing {k}; available: {sorted(d)}")
    return d[k]

SEED = 42
mlca_runtime.set_seed(SEED)
inp = mlca_runtime.inputs()
p = json.loads(inp["_protocol"].read_text())
targets = need(p, "targets") or []
excluded = need(p, "excluded_columns") or []
group = need(p, "group_column")
task = need(p, "task")
metric = task.get("primary_metric") or "silhouette"
df = pd.read_parquet(inp["train"])

drop = set(targets) | set(excluded) | ({group} if group else set())
cols = []
for c in df.columns:
    if c in drop:
        continue
    s = df[c]; nm = str(c).lower()
    if s.nunique(dropna=True) == len(s) and len(s) > 1:
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            continue
        if pd.api.types.is_integer_dtype(s) and any(t in nm for t in ("id", "code", "no", "num", "key")):
            continue
    cols.append(c)
if not cols:
    raise ValueError("no feature columns")
X = df[cols].copy(); n = len(X)
num = [c for c in cols if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
cat = [c for c in cols if c not in num]
width = len(num) + sum(X[c].nunique(dropna=False) for c in cat)
dense = width <= 300

def make_pre():
    tf = []
    if num:
        tf.append(("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler(with_mean=dense))]), num))
    if cat:
        tf.append(("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense))]), cat))
    pre = ColumnTransformer(tf, sparse_threshold=0.0 if dense else 0.3)
    if dense:
        pre.set_output(transform="pandas")
    return pre

kmax = min(8, n - 1)
rng = np.random.default_rng(SEED)
sub = rng.choice(n, size=10000, replace=False) if n > 10000 else np.arange(n)
take = lambda M, i: M.iloc[i] if hasattr(M, "iloc") else M[i]
rows, models = [], {}
for k in range(2, kmax + 1):
    pipe = Pipeline([("prep", make_pre()), ("cluster", MiniBatchKMeans(n_clusters=k, random_state=SEED, batch_size=min(1024, max(256, n // 10)), n_init="auto"))])
    fit = mlca_runtime.fit_record(pipe, X, partition="train")
    Z = fit.named_steps["prep"].transform(X)
    lab = fit.named_steps["cluster"].labels_.copy()
    Zs = take(Z, sub); ls_sub = lab[sub]
    if len(np.unique(lab)) > 1:
        sil = float(silhouette_score(Zs, ls_sub, random_state=SEED))
    else:
        sil = -1.0
    db = float(davies_bouldin_score(Zs, ls_sub))
    ch = float(calinski_harabasz_score(Zs, ls_sub))
    ar = []
    for b in range(2):
        w = rng.multinomial(n, np.full(n, 1.0 / n))
        km = MiniBatchKMeans(n_clusters=k, random_state=SEED + b, batch_size=min(1024, max(256, n // 10)), n_init="auto")
        mlca_runtime.fit_record(km, Z, partition="train", sample_weight=w)
        ar.append(adjusted_rand_score(lab, km.predict(Z)))
    rows.append([k, sil, db, ch, float(np.mean(ar))]); models[k] = fit

arr = np.array(rows, float)
score = rankdata(-arr[:, 1]) + rankdata(arr[:, 2]) + rankdata(-arr[:, 3]) + rankdata(-arr[:, 4])
best = rows[int(np.argmin(score))]
mlca_runtime.save_model(models[best[0]], "alternative_model.pkl")
mlca_runtime.record_metric(metric, float(best[1]), "train")
mlca_runtime.record_metric("n_clusters", float(best[0]), "train")
mlca_runtime.record_metric("stability", float(best[4]), "train")
print(f"rows={n} feats={len(cols)} dense={dense} k_range=2..{kmax}")
print(f"best_k={best[0]} silhouette={best[1]:.4f} stability={best[4]:.4f}")
