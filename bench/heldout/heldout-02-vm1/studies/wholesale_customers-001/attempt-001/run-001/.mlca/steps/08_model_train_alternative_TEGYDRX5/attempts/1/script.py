import mlca_runtime, json, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

targets = list(protocol.get('targets') or [])
excluded = list(protocol.get('excluded_columns') or [])
group = protocol.get('group_column')
n = len(train)

# Feature schema: drop targets, excluded, group, and key-like text columns
feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        train[c + '_ts'] = s.astype('int64') // 10**9
        val[c + '_ts'] = val[c].astype('int64') // 10**9
        feature_cols.append(c + '_ts')
    elif pd.api.types.is_numeric_dtype(s):
        feature_cols.append(c)
    else:
        if s.nunique(dropna=True) < n:  # key-like if all distinct
            if s.nunique(dropna=True) <= max(1, int(0.05 * n)):
                feature_cols.append(c)

num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]
X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()

def make_pipeline(k):
    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                         ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    trs = [('num', num_pipe, num_cols)]
    if cat_cols:
        trs.append(('cat', cat_pipe, cat_cols))
    pre = ColumnTransformer(trs, remainder='drop')
    pre.set_output(transform='pandas')
    pipe = Pipeline([('pre', pre), ('kmeans', MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10))])
    pipe.set_output(transform='pandas')
    return pipe

ks = list(range(2, 9))
rng = np.random.RandomState(42)
boot_w = [rng.multinomial(n, np.ones(n) / n).astype(float) for _ in range(5)]
records = []
for k in ks:
    pipe = make_pipeline(k)
    pipe = mlca_runtime.fit_record(pipe, X_train, 'train')
    pre_X = pipe.named_steps['pre'].transform(X_train)
    labels = pipe.named_steps['kmeans'].labels_
    sil = float(silhouette_score(pre_X, labels, sample_size=min(10000, n), random_state=42))
    db = float(davies_bouldin_score(pre_X, labels))
    ch = float(calinski_harabasz_score(pre_X, labels))
    aris = []
    for w in boot_w:
        bp = make_pipeline(k)
        bp = mlca_runtime.fit_record(bp, X_train, 'train', kmeans__sample_weight=w)
        aris.append(adjusted_rand_score(labels, bp.named_steps['kmeans'].labels_))
    st = float(np.mean(aris))
    records.append({'k': k, 'pipe': pipe, 'sil': sil, 'db': db, 'ch': ch, 'st': st})

stable = [r for r in records if r['st'] >= 0.5] or records
best = max(stable, key=lambda r: r['sil'])
pipe = best['pipe']

train_out = pd.DataFrame({'row_id': X_train.index, 'cluster': pipe.predict(X_train)})
mlca_runtime.save_row_outputs(train_out, 'clusters_train.parquet')
val_out = pd.DataFrame({'row_id': X_val.index, 'cluster': pipe.predict(X_val)})
mlca_runtime.save_row_outputs(val_out, 'clusters_val.parquet')

mlca_runtime.save_model(pipe, 'alternative_model.pkl')
mlca_runtime.record_metric('silhouette', best['sil'], 'train')
mlca_runtime.record_metric('davies_bouldin', best['db'], 'train')
mlca_runtime.record_metric('calinski_harabasz', best['ch'], 'train')
mlca_runtime.record_metric('stability_ari', best['st'], 'train')
mlca_runtime.record_metric('n_clusters', float(best['k']), 'train')

print(f"train rows={n} features={len(feature_cols)}")
print(f"candidate k={ks}")
print(f"selected k={best['k']} silhouette={best['sil']:.4f}")
print(f"stability_ari={best['st']:.4f}")
