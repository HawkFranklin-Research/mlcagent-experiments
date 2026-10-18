import mlca_runtime
import json, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
for key in ['targets', 'excluded_columns', 'task']:
    if key not in proto:
        raise KeyError(f"Missing {key}. Available: {list(proto.keys())}")
targets, excluded, task = proto['targets'], proto['excluded_columns'], proto['task']
group = proto.get('group_column')

df = pd.read_parquet(inp['train'])
n = len(df)
cand = [c for c in df.columns if c not in set(targets) | set(excluded) | ({group} if group else set())]
drop_keys = []
for c in cand:
    s = df[c]
    if s.dtype == object and s.nunique(dropna=True) >= 0.9 * n:
        drop_keys.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique() == n:
        d = s.sort_values().diff().dropna()
        if len(d) and (d == 1).all():
            drop_keys.append(c)
feature_cols = [c for c in cand if c not in drop_keys]
num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]
X = df[feature_cols].copy()

pre = ColumnTransformer(
    [('num', StandardScaler(), num_cols)] +
    ([('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)] if cat_cols else []),
    remainder='drop'
)
pre.set_output(transform='pandas')
pre_fitted = mlca_runtime.fit_record(pre, X, 'train')
X_scaled = pre_fitted.transform(X)

sub_n = min(5000, n)
best = None
for k in range(2, 11):
    km = mlca_runtime.fit_record(
        MiniBatchKMeans(n_clusters=k, random_state=42, n_init='auto', batch_size=1024),
        X_scaled, 'train'
    )
    labels = km.predict(X_scaled)
    sil = silhouette_score(X_scaled, labels, sample_size=sub_n, random_state=42)
    aris = []
    for b in range(3):
        mult = np.random.default_rng(100 + b).multinomial(n, [1.0 / n] * n)
        kmb = mlca_runtime.fit_record(
            MiniBatchKMeans(n_clusters=k, random_state=42, n_init='auto', batch_size=1024),
            X_scaled, 'train', sample_weight=mult
        )
        aris.append(adjusted_rand_score(labels, kmb.predict(X_scaled)))
    stab = float(np.mean(aris))
    score = sil + 0.3 * stab
    if best is None or score > best['score']:
        best = {'k': k, 'sil': sil, 'stab': stab, 'score': score}

pipe = Pipeline([
    ('pre', pre),
    ('km', MiniBatchKMeans(n_clusters=best['k'], random_state=42, n_init='auto', batch_size=1024))
])
pipe.set_output(transform='pandas')
pipe = mlca_runtime.fit_record(pipe, X, 'train')
mlca_runtime.save_model(pipe, "alternative_model.pkl")
mlca_runtime.record_metric("silhouette", float(best['sil']), "train")
mlca_runtime.record_metric("stability_ari", float(best['stab']), "train")
print(f"n={n} features={len(feature_cols)} k={best['k']} sil={best['sil']:.3f} stab={best['stab']:.3f} sub={sub_n}")
