import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

protocol = json.loads(inp["_protocol"].read_text())
task = protocol["task"]
targets = list(protocol.get("targets", []) or [])
excluded = list(protocol.get("excluded_columns", []) or [])
group = protocol.get("group_column")

train = pd.read_parquet(inp["train"])
print("train", train.shape)

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feat_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if s.dtype == object or str(s.dtype).startswith("string"):
        if s.nunique(dropna=True) > 0.05 * len(s):
            continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s) and s.nunique() > 1:
        continue
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    feat_cols.append(c)

num_cols, cat_cols = [], []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 10:
        num_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) <= 2:
        num_cols.append(c)
    else:
        cat_cols.append(c)
if len(cat_cols) > 0 and train[cat_cols].nunique().max() > 0.05 * len(train):
    cat_cols = [c for c in cat_cols if train[c].nunique(dropna=True) <= 0.05 * len(train)]
    num_cols = [c for c in feat_cols if c not in cat_cols]

X = train[feat_cols].copy()
print("feat", len(feat_cols), "num", len(num_cols), "cat", len(cat_cols))

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))

pre = ColumnTransformer(transformers, remainder="drop")
try:
    pre.set_output(transform="pandas")
except Exception:
    pass

k_range = list(range(2, 9))
results = []
n = len(X)
rng = np.random.RandomState(42)
sub_idx = rng.choice(n, size=min(10000, n), replace=False)

best = None
best_k = None
best_score = -np.inf

for k in k_range:
    pipe = Pipeline([("pre", pre), ("km", MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5, batch_size=1024))])
    fitted = mlca_runtime.fit_record(pipe, X, partition="train")
    Xt = fitted.named_steps["pre"].transform(X)
    if hasattr(Xt, "values"):
        Xt_arr = Xt.values
    else:
        Xt_arr = np.asarray(Xt)
    sub = Xt_arr[sub_idx]
    try:
        sil = silhouette_score(sub, fitted.named_steps["km"].predict(X[feat_cols].iloc[sub_idx]) if False else fitted.named_steps["km"].labels_[sub_idx], random_state=42)
    except Exception:
        sil = float("nan")
    try:
        db = davies_bouldin_score(sub, fitted.named_steps["km"].labels_[sub_idx])
    except Exception:
        db = float("nan")
    try:
        ch = calinski_harabasz_score(sub, fitted.named_steps["km"].labels_[sub_idx])
    except Exception:
        ch = float("nan")

    # bootstrap stability via sample-weight multiplicities
    stab_scores = []
    base_labels = fitted.named_steps["km"].labels_
    for b in range(3):
        w = np.random.RandomState(100 + b).multinomial(n, np.ones(n) / n, size=1)[0].astype(float)
        pipe_b = Pipeline([("pre", pre), ("km", MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=1024))])
        try:
            fb = mlca_runtime.fit_record(pipe_b, X, partition="train", sample_weight=w)
            Xb_t = fb.named_steps["pre"].transform(X)
            lb = fb.named_steps["km"].labels_
            # cluster agreement via contingency best match (label invariant)
            from scipy.optimize import linear_sum_assignment
            contingency = np.zeros((k, k))
            for a, c in zip(base_labels, lb):
                contingency[a, c] += 1
            row_ind, col_ind = linear_sum_assignment(-contingency)
            acc = contingency[row_ind, col_ind].sum() / n
            stab_scores.append(acc)
        except Exception:
            pass
    stab = float(np.mean(stab_scores)) if stab_scores else float("nan")

    results.append({"k": k, "silhouette": sil, "davies_bouldin": db,
                    "calinski_harabasz": ch, "stability": stab})
    score = sil if not np.isnan(sil) else -np.inf
    if score > best_score:
        best_score = score
        best_k = k
        best = fitted

print("k_range", k_range, "best_k", best_k)

res_df = pd.DataFrame(results).round(4)
mlca_runtime.save_table(res_df, "cluster_selection.csv", caption="Cluster count selection metrics")

mlca_runtime.save_model(best, "alternative_model.pkl")
mlca_runtime.record_metric("silhouette", float(best_score), "train")

# per-row outputs
labels = best.named_steps["km"].labels_
assign = pd.DataFrame({"row_id": train.index, "cluster": labels})
assign = assign.set_index("row_id")
mlca_runtime.save_row_outputs(assign, "train_clusters.parquet")
print("saved")
