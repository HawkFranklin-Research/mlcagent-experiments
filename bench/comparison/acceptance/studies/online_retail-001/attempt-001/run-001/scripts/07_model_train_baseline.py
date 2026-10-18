import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from pandas.api.types import (is_object_dtype, is_string_dtype, is_integer_dtype,
                              is_bool_dtype, is_numeric_dtype, is_datetime64_any_dtype)

mlca_runtime.set_seed(42)
rng = np.random.RandomState(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
train = pd.read_parquet(inp['train'])

drop = set(proto.get('targets') or []) | set(proto.get('excluded_columns') or [])
if proto.get('group_column'):
    drop.add(proto['group_column'])
n = len(train)
feat_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if is_datetime64_any_dtype(s):
        continue
    if (is_object_dtype(s) or is_string_dtype(s)) and s.nunique(dropna=False) == n:
        continue
    if is_integer_dtype(s) and not is_bool_dtype(s) and s.nunique(dropna=False) == n:
        continue
    feat_cols.append(c)

with open(inp['05_preprocess_fit_preprocessing_GGPQ6YHJ/preprocessor.pkl'], 'rb') as f:
    prep = pickle.load(f)
with open(inp['06_features_retain_features_JSGE8JGA/selector_all.pkl'], 'rb') as f:
    sel = pickle.load(f)

decl = getattr(prep, 'feature_names_in_', None)
if decl is not None:
    raw_cols = [c for c in decl if c in train.columns]
else:
    raw_cols = [c for c in feat_cols if is_numeric_dtype(train[c]) or
                is_object_dtype(train[c]) or is_string_dtype(train[c]) or
                str(train[c].dtype) == 'category']
X = train[raw_cols].copy()

use_prep = True
use_sel = sel is not None
if use_sel and hasattr(sel, 'feature_names_in_') and list(sel.feature_names_in_) == raw_cols:
    use_prep = False

def make(k, n_init=5):
    steps = []
    if use_prep:
        steps.append(('prep', prep))
    if use_sel:
        steps.append(('sel', sel))
    steps.append(('km', KMeans(n_clusters=k, n_init=n_init, random_state=42)))
    return Pipeline(steps)

def transform(f, Xd):
    Z = f[:-1].transform(Xd)
    return Z.toarray() if hasattr(Z, 'toarray') else Z

sub_n = min(10000, len(X))
sub_idx = rng.choice(len(X), size=sub_n, replace=False)
X_sub = X.iloc[sub_idx]

cands = [k for k in range(2, 9) if k < len(X)]
scored = []
for k in cands:
    f = mlca_runtime.fit_record(make(k), X, 'train')
    Z = transform(f, X_sub)
    lab = f.predict(X_sub)
    if len(np.unique(lab)) < 2:
        sil = -1.0
    else:
        sil = float(silhouette_score(Z, lab, sample_size=min(sub_n, 5000), random_state=42))
    scored.append((sil, k, f))
scored.sort(key=lambda t: -t[0])

aris = {}
for sil, k, f in scored[:2]:
    Z_all = transform(f, X)
    Z_sub = transform(f, X_sub)
    base = f.predict(X_sub)
    vals = []
    for b in range(3):
        w = rng.multinomial(len(X), np.ones(len(X)) / len(X)).astype(float)
        km_b = mlca_runtime.fit_record(
            KMeans(n_clusters=k, n_init=3, random_state=42 + b),
            pd.DataFrame(Z_all), 'train', sample_weight=w)
        vals.append(float(adjusted_rand_score(base, km_b.predict(pd.DataFrame(Z_sub)))))
    aris[k] = float(np.mean(vals))

pick = max(scored, key=lambda t: 0.6 * t[0] + 0.4 * aris.get(t[1], 0.0))
sil, bk, _ = pick
final = mlca_runtime.fit_record(make(bk, n_init=20), X, 'train')
mlca_runtime.save_model(final, 'model.pkl')
mlca_runtime.record_metric('silhouette', float(sil), 'train')
mlca_runtime.record_metric('cluster_stability_ari', aris.get(bk, 0.0), 'train')
print(f"rows={len(X)} features={len(raw_cols)} clusters={bk} silhouette={sil:.4f} ari={aris.get(bk,0.0):.4f}")
