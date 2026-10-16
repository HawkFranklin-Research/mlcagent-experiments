import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
drop.add("InvoiceDate")

feat_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
            continue
    else:
        if s.nunique(dropna=True) == len(s):
            continue
    feat_cols.append(c)

X = train[feat_cols].copy()
num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in feat_cols if c not in num_cols]
n_enc = sum(int(X[c].nunique(dropna=True)) + 1 for c in cat_cols)
wide = (len(num_cols) + n_enc) > 300


def build_pipe(k, seed):
    trs = []
    if num_cols:
        trs.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        trs.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                         sparse_output=wide), cat_cols))
    pre = ColumnTransformer(trs)
    if not wide:
        pre.set_output(transform="pandas")
    return Pipeline([("pre", pre),
                     ("km", MiniBatchKMeans(n_clusters=k, random_state=seed,
                                            n_init=3, batch_size=1024))])


k_range = list(range(2, 11))
rows, fitted_models = [], {}
for k in k_range:
    fitted = mlca_runtime.fit_record(build_pipe(k, 42), X, "train")
    Xt = fitted.named_steps["pre"].transform(X)
    labels = fitted.named_steps["km"].labels_
    n = Xt.shape[0]
    if n > 10000:
        idx = np.random.RandomState(42).choice(n, 10000, replace=False)
        Xs, ls = Xt[idx], labels[idx]
    else:
        Xs, ls = Xt, labels
    if hasattr(Xs, "toarray"):
        Xs = Xs.toarray()
    else:
        Xs = np.asarray(Xs)
    rows.append({
        "k": k,
        "silhouette": float(silhouette_score(Xs, ls, random_state=42)),
        "davies_bouldin": float(davies_bouldin_score(Xs, ls)),
        "calinski_harabasz": float(calinski_harabasz_score(Xs, ls)),
    })
    fitted_models[k] = fitted

rng = np.random.RandomState(42)
n = len(X)
stab = {}
for k in k_range:
    ref = fitted_models[k].named_steps["km"].labels_
    aris = []
    for b in range(5):
        idx = rng.randint(0, n, size=n)
        counts = np.bincount(idx, minlength=n).astype(float)
        fitted = mlca_runtime.fit_record(build_pipe(k, 100 + b), X, "train",
                                         km__sample_weight=counts)
        aris.append(float(adjusted_rand_score(ref,
                                              fitted.named_steps["km"].labels_)))
    stab[k] = float(np.mean(aris))

for r in rows:
    r["stability"] = stab[r["k"]]
    r["score"] = r["silhouette"] + 0.5 * r["stability"]

best = max(rows, key=lambda r: r["score"])
best_k = best["k"]
best_model = fitted_models[best_k]

mlca_runtime.save_model(best_model, "alternative_model.pkl")
mlca_runtime.record_metric("silhouette", best["silhouette"], "train")
mlca_runtime.record_metric("stability", best["stability"], "train")
mlca_runtime.record_metric("n_clusters", float(best_k), "train")

summary = pd.DataFrame(rows).round(4)
mlca_runtime.save_table(summary, "k_selection.csv", "Cluster count selection")

print(f"features: {len(feat_cols)} (num={len(num_cols)}, cat={len(cat_cols)}), wide={wide}")
print(f"best k={best_k}, silhouette={best['silhouette']:.4f}, stability={best['stability']:.4f}")
