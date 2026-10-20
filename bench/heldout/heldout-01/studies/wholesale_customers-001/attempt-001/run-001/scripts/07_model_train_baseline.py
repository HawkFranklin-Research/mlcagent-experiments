import mlca_runtime
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_Q63KMMVN/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_retain_features_S31FQA6H/selector_all.pkl'], 'rb') as f:
    sel = pickle.load(f)

def _n_in(obj):
    if hasattr(obj, 'n_features_in_'):
        return int(obj.n_features_in_)
    fi = getattr(obj, 'feature_names_in_', None)
    return len(fi) if fi is not None else None

pre_n_in, sel_n_in = _n_in(pre), _n_in(sel)

# Decide frontend: if sel accepts the same raw dims as pre, sel already wraps pre.
if sel_n_in is not None and pre_n_in is not None and sel_n_in == pre_n_in:
    frontend = sel
else:
    frontend = Pipeline([('pre', pre), ('sel', sel)])

first_step = frontend.named_steps[list(frontend.named_steps)[0]] if isinstance(frontend, Pipeline) else frontend
fi = getattr(first_step, 'feature_names_in_', None)
feature_cols = list(fi) if fi is not None else list(train.columns)
X_train = train[feature_cols].copy()

if isinstance(frontend, Pipeline):
    pre_steps = list(frontend.steps)
else:
    pre_steps = [('fe', frontend)]

candidates = list(range(2, 9))
n = len(X_train)
rng = np.random.RandomState(42)
fitted_by_k, results = {}, []

for k in candidates:
    pipe = Pipeline(pre_steps + [('km', KMeans(n_clusters=k, n_init=10, random_state=42))])
    model = mlca_runtime.fit_record(pipe, X_train, 'train')
    labels = model.predict(X_train)
    Xt = model[:-1].transform(X_train)
    Xt_arr = Xt.values if hasattr(Xt, 'values') else Xt
    try:
        ss = float(silhouette_score(Xt_arr, labels, sample_size=min(10000, n), random_state=42))
    except Exception:
        ss = float('nan')
    try:
        db = float(davies_bouldin_score(Xt_arr, labels))
    except Exception:
        db = float('nan')
    try:
        ch = float(calinski_harabasz_score(Xt_arr, labels))
    except Exception:
        ch = float('nan')
    aris = []
    for b in range(3):
        w = rng.multinomial(n, np.ones(n) / n).astype(float)
        boot_pipe = Pipeline(pre_steps + [('km', KMeans(n_clusters=k, n_init=10, random_state=100 + b))])
        bf = mlca_runtime.fit_record(boot_pipe, X_train, 'train', **{'km__sample_weight': w})
        bl = bf.predict(X_train)
        aris.append(float(adjusted_rand_score(labels, bl)))
    stability = float(np.mean(aris)) if aris else float('nan')
    results.append({'k': k, 'silhouette': ss, 'db': db, 'ch': ch, 'stability': stability})
    fitted_by_k[k] = model

df = pd.DataFrame(results)
for col in ['silhouette', 'db', 'ch', 'stability']:
    vals = df[col].values.astype(float)
    mu, sd = np.nanmean(vals), np.nanstd(vals)
    df[col + '_z'] = (vals - mu) / sd if sd > 0 else 0.0
df['score'] = df['silhouette_z'] - df['db_z'] + df['ch_z'] + df['stability_z']
best_row = df.loc[df['score'].idxmax()]
best_k = int(best_row['k'])
final_model = fitted_by_k[best_k]

mlca_runtime.save_model(final_model, 'model.pkl')
mlca_runtime.record_metric('silhouette', float(best_row['silhouette']), 'train')
mlca_runtime.record_metric('best_k', float(best_k), 'train')
mlca_runtime.record_metric('stability', float(best_row['stability']), 'train')

print(f"Selected k={best_k} from candidates {candidates}")
print(f"Silhouette={best_row['silhouette']:.4f} DB={best_row['db']:.4f} CH={best_row['ch']:.2f} stability={best_row['stability']:.4f}")
print(f"Train rows: {n}, features: {len(feature_cols)}")
