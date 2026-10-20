import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
task = protocol['task']
primary = task['primary_metric']
targets = list(protocol.get('targets', []))
excluded = list(protocol.get('excluded_columns', []))
group_col = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
n = len(train)
feat_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=True) >= n:
            continue
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=True) >= n and n > 1:
            continue
    feat_cols.append(c)

with open(inputs['07_model_train_baseline_9CYH36JK/model.pkl'], 'rb') as f:
    baseline = pickle.load(f)
with open(inputs['08_model_train_alternative_R5ZEQMH2/alternative_model.pkl'], 'rb') as f:
    alternative = pickle.load(f)
candidates = [('baseline', baseline), ('alternative', alternative)]

def inputs_for(model, df):
    if hasattr(model, 'feature_names_in_'):
        cols = list(model.feature_names_in_)
        X = df[cols].copy()
    else:
        X = df[feat_cols].copy()
    if isinstance(model, Pipeline) and len(model.steps) > 1:
        Xt = model[:-1].transform(X)
    else:
        Xt = X
    return X, Xt

def metrics(Xt, labels):
    if isinstance(Xt, pd.DataFrame):
        Xt = Xt.values
    Xt = np.asarray(Xt)
    nl = len(np.unique(labels))
    if nl < 2 or nl >= len(labels):
        return float('nan'), float('nan'), float('nan')
    try:
        if len(Xt) > 10000:
            sil = float(silhouette_score(Xt, labels, sample_size=10000, random_state=42))
        else:
            sil = float(silhouette_score(Xt, labels))
    except Exception:
        sil = float('nan')
    try:
        ch = float(calinski_harabasz_score(Xt, labels))
    except Exception:
        ch = float('nan')
    try:
        db = float(davies_bouldin_score(Xt, labels))
    except Exception:
        db = float('nan')
    return sil, ch, db

def r4(v):
    if v is None:
        return float('nan')
    try:
        f = float(v)
    except Exception:
        return float('nan')
    if np.isnan(f):
        return float('nan')
    return float(f"{f:.4g}")

rows = []
sil_by = {}
for name, model in candidates:
    X, Xt = inputs_for(model, val)
    labels = np.asarray(model.predict(X))
    sil, ch, db = metrics(Xt, labels)
    sil_by[name] = sil
    for m, v in [('silhouette', sil), ('calinski_harabasz', ch), ('davies_bouldin', db)]:
        rows.append({'model': name, 'metric': m, 'value': r4(v)})

mdf = pd.DataFrame(rows)
mlca_runtime.save_table(mdf, 'validation_metrics.csv', caption='Internal cluster metrics on validation partition')

valid = {k: v for k, v in sil_by.items() if not np.isnan(v)}
if valid:
    best = max(valid, key=lambda k: valid[k])
    best_score = float(valid[best])
    mlca_runtime.record_metric(primary, best_score, 'val')
    mlca_runtime.record_fact_candidate('best_validation_score', r4(best_score), source='validation_metrics.csv')
    mlca_runtime.record_fact_candidate('best_model', best, source='validation_metrics.csv')
    print('selected:', best, 'val', primary, '=', r4(best_score))
print('features:', len(feat_cols), 'val rows:', len(val))
