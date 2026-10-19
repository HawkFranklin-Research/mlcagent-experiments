import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
targets = proto['targets']; excluded = proto['excluded_columns']
group = proto.get('group_column'); top_keys = list(proto.keys())
train = pd.read_parquet(inp['train'])
with open(inp['05_preprocess_fit_preprocessing_MATQH9FA/preprocessor.pkl'], 'rb') as f: pre = pickle.load(f)
with open(inp['06_features_retain_features_5RGESHAY/selector_all.pkl'], 'rb') as f: sel = pickle.load(f)

def first_in(obj):
    if isinstance(obj, Pipeline) and obj.named_steps:
        return getattr(list(obj.named_steps.values())[0], 'feature_names_in_', None)
    return getattr(obj, 'feature_names_in_', None)

pre_in, sel_in = first_in(pre), first_in(sel)
if pre_in is not None and sel_in is not None and set(pre_in) == set(sel_in):
    base = sel; first = sel_in
else:
    base = Pipeline([('pre', pre), ('sel', sel)])
    first = pre_in if pre_in is not None else sel_in

if first is not None:
    miss = [c for c in first if c not in train.columns]
    if miss: raise ValueError(f"Missing {miss[:5]}; top keys: {top_keys}")
    X_train = train[list(first)]
else:
    drop = set(list(targets) + list(excluded) + ([group] if group else []))
    X_train = train[[c for c in train.columns if c not in drop]]

try: base.set_output(transform='pandas')
except Exception: pass
base_fitted = mlca_runtime.fit_record(base, X_train, partition='train')
Xt = base_fitted.transform(X_train)
if isinstance(Xt, pd.DataFrame):
    Xt_df = Xt
elif hasattr(Xt, 'toarray'):
    Xt_df = pd.DataFrame(Xt.toarray(), index=X_train.index)
else:
    Xt_df = pd.DataFrame(np.asarray(Xt), index=X_train.index)

n = len(Xt_df); k_grid = list(range(2, 11))
rng = np.random.RandomState(42)
sidx = rng.choice(n, 10000, replace=False) if n > 10000 else np.arange(n)
Xs = Xt_df.iloc[sidx]

fitted_kms, rows, stabs = {}, [], {}
for k in k_grid:
    fk = mlca_runtime.fit_record(
        KMeans(n_clusters=k, n_init=10, random_state=42), Xt_df, partition='train')
    ls = fk.labels_[sidx]
    sil = silhouette_score(Xs, ls, random_state=42)
    db = davies_bouldin_score(Xs, ls)
    ch = calinski_harabasz_score(Xs, ls)
    fitted_kms[k] = fk
    rows.append({'k': k, 'silhouette': round(sil, 4),
                 'davies_bouldin': round(db, 4),
                 'calinski_harabasz': round(ch, 4)})

n_boot = 2
for k in k_grid:
    bl = fitted_kms[k].labels_; aris = []
    for b in range(n_boot):
        r = np.random.RandomState(1000 + b)
        w = r.multinomial(n, np.ones(n) / n).astype(float)
        fb = mlca_runtime.fit_record(
            KMeans(n_clusters=k, n_init=10, random_state=42),
            Xt_df, partition='train', sample_weight=w)
        aris.append(adjusted_rand_score(bl, fb.labels_))
    stabs[k] = float(np.mean(aris))

scores = {r['k']: r['silhouette'] * (0.5 + 0.5 * stabs[r['k']]) for r in rows}
best_k = max(scores, key=scores.get)
mlca_runtime.save_model(Pipeline([('base', base_fitted), ('km', fitted_kms[best_k])]), 'model.pkl')

brow = next(r for r in rows if r['k'] == best_k)
mlca_runtime.record_metric('silhouette', brow['silhouette'], 'train')
mlca_runtime.record_metric('davies_bouldin', brow['davies_bouldin'], 'train')
mlca_runtime.record_metric('calinski_harabasz', brow['calinski_harabasz'], 'train')
mlca_runtime.record_metric('n_clusters', int(best_k), 'train')
mlca_runtime.record_metric('stability_ari', stabs[best_k], 'train')

print(f"rows={n} cols={Xt_df.shape[1]}")
print('k/sil: ' + ' '.join(f"{r['k']}:{r['silhouette']:.3f}" for r in rows))
print(f"best k={best_k} sil={brow['silhouette']} ari={stabs[best_k]:.3f}")
