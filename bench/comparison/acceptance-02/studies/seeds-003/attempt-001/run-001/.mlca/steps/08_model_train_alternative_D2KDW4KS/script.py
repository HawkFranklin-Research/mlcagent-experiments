import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score,
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in proto:
        raise KeyError(f"Missing {key}. Available top-level keys: {sorted(proto)}")

targets = set(proto["targets"])
excluded = set(proto["excluded_columns"])
group = proto.get("group_column")
train = pd.read_parquet(inp["train"])
n = len(train)

def is_keylike(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=True) == s.notna().sum()
    if pd.api.types.is_integer_dtype(s):
        vals = pd.Series(s.dropna().unique()).sort_values()
        return len(vals) == len(s) and len(vals) > 1 and (vals.diff().dropna() == 1).all()
    return False

feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group and not is_keylike(train[c])
]
numeric_cols, cat_cols, high_card = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=True) > max(5, 0.05 * n):
            high_card.append(c)
        else:
            cat_cols.append(c)
    else:
        high_card.append(c)

width = len(numeric_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
transformers = [("num", num_pipe, numeric_cols)]
if cat_cols:
    transformers.append((
        "cat",
        OneHotEncoder(handle_unknown="ignore", sparse_output=(width > 300)),
        cat_cols,
    ))
pre = ColumnTransformer(transformers, remainder="drop")
if width <= 300:
    pre.set_output(transform="pandas")

def make_pipe(k):
    pre_k = clone(pre)
    if width <= 300:
        pre_k.set_output(transform="pandas")
    pipe = Pipeline([
        ("pre", pre_k),
        ("kmeans", MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
            batch_size=min(256, max(2, n)),
        )),
    ])
    if width <= 300:
        pipe.set_output(transform="pandas")
    return pipe

X = train[feature_cols].copy()
rng = np.random.default_rng(42)
ks = range(2, max(2, min(8, n - 1)) + 1)
results = {}
for k in ks:
    pipe = mlca_runtime.fit_record(make_pipe(k), X, "train")
    labels = pipe.predict(X)
    Xt = pipe.named_steps["pre"].transform(X)
    sil = silhouette_score(Xt, labels, sample_size=min(10000, n), random_state=42)
    db = davies_bouldin_score(Xt, labels)
    ch = calinski_harabasz_score(Xt, labels)
    aris = []
    for _ in range(5):
        weights = rng.multinomial(n, np.ones(n) / n)
        boot = mlca_runtime.fit_record(make_pipe(k), X, "train", kmeans__sample_weight=weights)
        aris.append(adjusted_rand_score(labels, boot.predict(X)))
    results[k] = (float(sil), float(db), float(ch), float(np.mean(aris)), pipe)

best_k = max(results, key=lambda k: results[k][0] + 0.1 * results[k][3])
best_sil, best_db, best_ch, best_stab, best_pipe = results[best_k]
mlca_runtime.save_model(best_pipe, "alternative_model.pkl")
mlca_runtime.record_metric("silhouette", best_sil, "train")
mlca_runtime.record_metric("bootstrap_ari", best_stab, "train")
mlca_runtime.record_metric("n_clusters", float(best_k), "train")
print(f"rows={n} features={len(feature_cols)}")
print(f"selected_k={best_k} silhouette={best_sil:.4f} stability={best_stab:.4f}")
