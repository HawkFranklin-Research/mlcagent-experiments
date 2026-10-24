import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
task = protocol["task"]

train = pd.read_parquet(inputs["train"])
n_train = len(train)
drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feat_cols = [c for c in train.columns if c not in drop]
feat_cols = [c for c in feat_cols if not ((train[c].dtype.kind in "iu") and train[c].nunique() == n_train)]
feat_cols = [c for c in feat_cols if not (train[c].dtype == object and train[c].nunique() == n_train)]
feat_cols = [c for c in feat_cols if not pd.api.types.is_datetime64_any_dtype(train[c])]

num_cols, cat_cols = [], []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    elif s.nunique() < n_train and s.nunique() <= max(5, int(0.05 * n_train)):
        cat_cols.append(c)
feat_cols = num_cols + cat_cols

est_width = len(num_cols) + sum(int(train[c].nunique()) for c in cat_cols)
use_pandas_out = est_width <= 300

def make_pre():
    trs = []
    if num_cols:
        tn = Pipeline([("sc", StandardScaler())])
        if use_pandas_out:
            try: tn.set_output(transform="pandas")
            except Exception: pass
        trs.append(("num", tn, num_cols))
    if cat_cols:
        tc = Pipeline([("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        if use_pandas_out:
            try: tc.set_output(transform="pandas")
            except Exception: pass
        trs.append(("cat", tc, cat_cols))
    pre = ColumnTransformer(trs)
    if use_pandas_out:
        try: pre.set_output(transform="pandas")
        except Exception: pass
    return pre

def make_pipe(k, seed=42):
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10, batch_size=1024, max_iter=200)
    return Pipeline([("pre", make_pre()), ("km", km)])

X_train = train[feat_cols].copy()
pre_fit = mlca_runtime.fit_record(make_pre(), X_train, "train")
Xt = pre_fit.transform(X_train)
Xt_arr = Xt.values if hasattr(Xt, "values") else np.asarray(Xt)
if hasattr(Xt_arr, "toarray"):
    Xt_arr = Xt_arr.toarray()
Xt_arr = np.asarray(Xt_arr, dtype=float)

sil_n = min(10000, len(Xt_arr))
rng = np.random.RandomState(42)
sil_idx = rng.choice(len(Xt_arr), size=sil_n, replace=False) if len(Xt_arr) > sil_n else np.arange(len(Xt_arr))
sil_X = Xt_arr[sil_idx]

res = []
for k in range(2, 9):
    pipe_fit = mlca_runtime.fit_record(make_pipe(k), X_train, "train")
    lab_arr = np.asarray(pipe_fit.predict(X_train))
    if len(np.unique(lab_arr)) < 2:
        continue
    sil = float(silhouette_score(sil_X, lab_arr[sil_idx], random_state=42))
    aris = []
    for s in range(5):
        rng_b = np.random.RandomState(2000 + s)
        idx_b = rng_b.choice(n_train, size=n_train, replace=True)
        w = np.bincount(idx_b, minlength=n_train).astype(float)
        pf = mlca_runtime.fit_record(make_pipe(k), X_train, "train", sample_weight=w)
        lb = np.asarray(pf.predict(X_train))
        aris.append(float(adjusted_rand_score(lab_arr, lb)))
    stab = float(np.mean(aris)) if aris else 0.0
    composite = sil * max(stab, 0.0)
    res.append({"k": k, "silhouette": sil, "stability": stab, "composite": composite})
    mlca_runtime.record_metric(f"silhouette_k{k}", sil, "train")
    mlca_runtime.record_metric(f"stability_k{k}", stab, "train")

if not res:
    raise RuntimeError("No valid k produced >=2 clusters on training data")

best = max(res, key=lambda r: r["composite"])
chosen_k = best["k"]
final_fit = mlca_runtime.fit_record(make_pipe(chosen_k), X_train, "train")
mlca_runtime.record_metric("silhouette", best["silhouette"], "train")
mlca_runtime.record_metric("stability", best["stability"], "train")
mlca_runtime.record_metric("chosen_k", float(chosen_k), "train")
mlca_runtime.save_model(final_fit, "alternative_model.pkl")
print(f"tried k={[r['k'] for r in res]}, chosen_k={chosen_k}, sil={best['silhouette']:.4f}, stab={best['stability']:.4f}")
print(f"n_train={n_train}, n_features={len(feat_cols)} (num={len(num_cols)}, cat={len(cat_cols)})")
