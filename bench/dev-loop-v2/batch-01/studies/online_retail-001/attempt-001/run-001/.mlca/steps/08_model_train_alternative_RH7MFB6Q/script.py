import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
need = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in need if k not in protocol]
if missing:
    raise ValueError(f"Missing keys {missing}; available={list(protocol)}")

targets = set(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group = protocol["group_column"]
df = pd.read_parquet(inp["train"])
base_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group]
n = len(df)

num_cols, cat_cols, date_cols, dropped = [], [], [], []
for c in base_cols:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_integer_dtype(s) and s.notna().sum() > 1 and s.nunique(dropna=True) == s.notna().sum():
            dropped.append(c)
        else:
            num_cols.append(c)
    else:
        try:
            d = pd.to_datetime(s, errors="coerce")
            if d.notna().mean() > 0.8:
                date_cols.append(c)
                continue
        except Exception:
            pass
        u = s.nunique(dropna=True)
        if u == 0 or u / max(n, 1) > 0.05:
            dropped.append(c)
        else:
            cat_cols.append(c)

feature_cols = num_cols + cat_cols + date_cols
if not feature_cols:
    raise ValueError("No feature columns remain")
X = df[feature_cols].copy()
width = len(num_cols) + len(date_cols) + sum(df[c].nunique(dropna=True) for c in cat_cols)
dense = width <= 300
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)

transformers = []
if num_cols:
    transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols))
if cat_cols:
    transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)]), cat_cols))
for c in date_cols:
    transformers.append((f"date_{c}", Pipeline([
        ("dt", FunctionTransformer(pd.to_datetime, kw_args={"errors": "coerce"}, feature_names_out="one-to-one")),
        ("to_num", FunctionTransformer(pd.to_numeric, feature_names_out="one-to-one")),
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]), [c]))

pre = ColumnTransformer(transformers, remainder="drop")
if dense:
    pre.set_output(transform="pandas")

ks = [k for k in range(2, min(9, max(3, n // 20 + 2))) if 2 * k <= n] or [2]
rng = np.random.default_rng(42)
best_score, best = -np.inf, None
for k in ks:
    pipe = Pipeline([("pre", pre), ("kmeans", MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10, batch_size=1024, max_iter=100))])
    if dense:
        pipe.set_output(transform="pandas")
    pipe = mlca_runtime.fit_record(pipe, X, partition="train")
    Xt = pipe.named_steps["pre"].transform(X)
    names = pipe.named_steps["pre"].get_feature_names_out()
    if isinstance(Xt, pd.DataFrame):
        Xt_df = Xt
    elif hasattr(Xt, "toarray"):
        Xt_df = pd.DataFrame(Xt.toarray(), index=X.index, columns=names) if dense else pd.DataFrame.sparse.from_spmatrix(Xt, index=X.index, columns=names)
    else:
        Xt_df = pd.DataFrame(Xt, index=X.index, columns=names)

    labels = pipe.named_steps["kmeans"].predict(Xt_df)
    idx = rng.choice(n, size=min(10000, n), replace=False)
    Xs = Xt_df.iloc[idx]
    if hasattr(Xs, "sparse"):
        Xs = Xs.sparse.to_dense()
    sil = float(silhouette_score(Xs, labels[idx], random_state=42))

    aris = []
    for b in range(3):
        counts = np.random.default_rng(1000 + b).multinomial(n, np.full(n, 1.0 / n))
        kmb = MiniBatchKMeans(n_clusters=k, random_state=1000 + b, n_init=5, batch_size=1024)
        kmb = mlca_runtime.fit_record(kmb, Xt_df, partition="train", sample_weight=counts)
        aris.append(adjusted_rand_score(labels, kmb.predict(Xt_df)))
    stab = float(np.mean(aris))
    score = sil + 0.1 * stab
    mlca_runtime.record_metric(f"silhouette_k{k}", sil, "train")
    mlca_runtime.record_metric(f"stability_k{k}", stab, "train")
    if score > best_score:
        best_score, best = score, (k, pipe, sil, stab)

k_best, pipe_best, sil_best, stab_best = best
mlca_runtime.record_metric("selected_k", int(k_best), "train")
mlca_runtime.record_metric("selected_silhouette", float(sil_best), "train")
mlca_runtime.record_metric("selected_stability", float(stab_best), "train")
mlca_runtime.save_model(pipe_best, "alternative_model.pkl")
print(f"n={n} features={len(feature_cols)} width={width} k={k_best} sil={sil_best:.3f} stab={stab_best:.3f}")
