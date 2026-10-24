import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = list(protocol.get("targets") or [])
excluded = list(protocol.get("excluded_columns") or [])
group_col = protocol.get("group_column")
task = protocol["task"]

train = pd.read_parquet(inputs["train"])
n = len(train)
drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
rare = max(5, int(0.05 * n))
num_cols, cat_cols = [], []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        if s.dtype.kind in "iu" and s.nunique() == n:
            continue
        num_cols.append(c)
    else:
        if s.nunique() > rare:
            continue
        cat_cols.append(c)
feat_cols = num_cols + cat_cols
if not feat_cols:
    raise RuntimeError("No usable feature columns")

width = len(num_cols) + sum(int(train[c].nunique()) for c in cat_cols)
use_pandas = width <= 300

def make_pre():
    trs = []
    if num_cols:
        trs.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        trs.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    pre = ColumnTransformer(trs)
    if use_pandas:
        try:
            pre.set_output(transform="pandas")
        except Exception:
            pass
    return pre

def make_pipe(k, seed=42):
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10,
                         batch_size=1024, max_iter=200)
    return Pipeline([("pre", make_pre()), ("km", km)])

X = train[feat_cols].copy()
ref = mlca_runtime.fit_record(make_pre(), X, "train").transform(X)
ref_arr = np.asarray(ref.toarray() if hasattr(ref, "toarray") else ref, dtype=float)
sub_n = min(10000, len(ref_arr))
sil_idx = np.random.RandomState(42).choice(len(ref_arr), size=sub_n, replace=False) if len(ref_arr) > sub_n else np.arange(len(ref_arr))
sil_X = ref_arr[sil_idx]

rows = []
for k in range(2, 9):
    fitted = mlca_runtime.fit_record(make_pipe(k), X, "train")
    lab = np.asarray(fitted.predict(X))
    if len(np.unique(lab)) < 2:
        continue
    sil = float(silhouette_score(sil_X, lab[sil_idx], random_state=42))
    aris = []
    for s in range(5):
        idx = np.random.RandomState(2000 + s).choice(n, size=n, replace=True)
        w = np.bincount(idx, minlength=n).astype(float)
        fb = mlca_runtime.fit_record(make_pipe(k), X, "train", km__sample_weight=w)
        aris.append(float(adjusted_rand_score(lab, np.asarray(fb.predict(X)))))
    stab = float(np.mean(aris)) if aris else 0.0
    rows.append({"k": k, "silhouette": sil, "stability": stab,
                 "composite": sil * max(stab, 0.0)})
    mlca_runtime.record_metric(f"silhouette_k{k}", sil, "train")
    mlca_runtime.record_metric(f"stability_k{k}", stab, "train")

if not rows:
    raise RuntimeError("No valid cluster count produced >=2 clusters on training rows")
best = max(rows, key=lambda r: r["composite"])
chosen_k = int(best["k"])
final = mlca_runtime.fit_record(make_pipe(chosen_k), X, "train")

mlca_runtime.record_metric("silhouette", float(best["silhouette"]), "train")
mlca_runtime.record_metric("stability", float(best["stability"]), "train")
mlca_runtime.record_metric("chosen_k", float(chosen_k), "train")
mlca_runtime.save_model(final, "alternative_model.pkl")
print(f"silhouette subsample={sub_n} of {n}")
print(f"tried k={[r['k'] for r in rows]}, chosen_k={chosen_k}")
print(f"sil={best['silhouette']:.4f}, stability={best['stability']:.4f}, features={len(feat_cols)}")
