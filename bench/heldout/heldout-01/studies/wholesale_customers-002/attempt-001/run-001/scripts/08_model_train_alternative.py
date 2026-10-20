import json
import numpy as np
import pandas as pd
import mlca_runtime
from scipy.optimize import linear_sum_assignment
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
task = protocol["task"]
targets = list(protocol.get("targets") or [])
excluded = list(protocol.get("excluded_columns") or [])
group = protocol.get("group_column")

train = pd.read_parquet(inp["train"])
drop = set(targets) | set(excluded)
if group:
    drop.add(group)

feat_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if s.dtype == object or str(s.dtype).startswith("string"):
        if s.nunique(dropna=True) > 0.05 * len(s):
            continue
    if pd.api.types.is_integer_dtype(s) and s.notna().all() and s.nunique() == len(s) and len(s) > 1:
        continue
    feat_cols.append(c)

num_cols, cat_cols = [], []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    elif s.nunique(dropna=True) <= max(50, int(0.05 * len(train))):
        cat_cols.append(c)

X = train[feat_cols].copy()
print("train", train.shape, "feat", len(feat_cols), "num", len(num_cols), "cat", len(cat_cols))

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
pre = ColumnTransformer(transformers, remainder="drop")
pre.set_output(transform="pandas")

n = len(X)
rng = np.random.RandomState(42)
sub_idx = rng.choice(n, size=min(10000, n), replace=False)
print("diag subsample", len(sub_idx))

k_range = list(range(2, 9))
results = []
best = None
best_k = None
best_score = -np.inf

for k in k_range:
    pipe = Pipeline([("pre", pre), ("km", MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5, batch_size=1024))])
    fitted = mlca_runtime.fit_record(pipe, X, partition="train")
    base_labels = fitted.named_steps["km"].labels_
    Xt = np.asarray(fitted.named_steps["pre"].transform(X))
    sub_X = Xt[sub_idx]
    sub_lab = base_labels[sub_idx]
    sil = float(silhouette_score(sub_X, sub_lab, random_state=42)) if len(np.unique(sub_lab)) > 1 else float("nan")
    db = float(davies_bouldin_score(sub_X, sub_lab)) if len(np.unique(sub_lab)) > 1 else float("nan")
    ch = float(calinski_harabasz_score(sub_X, sub_lab)) if len(np.unique(sub_lab)) > 1 else float("nan")

    stab_scores = []
    for b in range(3):
        w = np.random.RandomState(100 + b).multinomial(n, np.ones(n) / n, size=1)[0].astype(float)
        try:
            fb = mlca_runtime.fit_record(
                Pipeline([("pre", pre), ("km", MiniBatchKMeans(n_clusters=k, random_state=42 + b, n_init=3, batch_size=1024))]),
                X, partition="train", km__sample_weight=w)
            lb = fb.named_steps["km"].labels_
            cont = np.zeros((k, k))
            for a, c in zip(base_labels, lb):
                cont[a, c] += 1
            r, cind = linear_sum_assignment(-cont)
            stab_scores.append(cont[r, cind].sum() / n)
        except Exception:
            pass
    stab = float(np.mean(stab_scores)) if stab_scores else float("nan")

    results.append({"k": k, "silhouette": sil, "davies_bouldin": db,
                    "calinski_harabasz": ch, "stability": stab})
    score = sil if (sil == sil) else -np.inf
    if score > best_score:
        best_score, best_k, best = score, k, fitted

print("best_k", best_k, "silhouette", round(best_score, 4))
for r in results:
    mlca_runtime.record_metric(f"silhouette_k{r['k']}", float(r["silhouette"]), "train")
    mlca_runtime.record_metric(f"stability_k{r['k']}", float(r["stability"]), "train")
mlca_runtime.record_metric("silhouette", float(best_score), "train")

mlca_runtime.save_model(best, "alternative_model.pkl")
print("saved alternative_model.pkl")
