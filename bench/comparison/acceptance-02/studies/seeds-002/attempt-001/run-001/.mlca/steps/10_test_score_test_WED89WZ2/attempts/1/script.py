import mlca_runtime
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

try:
    task = protocol['task']
    primary = task['primary_metric']
    targets = protocol['targets']
    excluded = protocol['excluded_columns']
    group_col = protocol['group_column']
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available: {list(protocol)}")

train = pd.read_parquet(inputs['train'])
test = pd.read_parquet(inputs['test'])

def key_like(s):
    if s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=True) == len(s) and len(s) > 0
    if pd.api.types.is_integer_dtype(s):
        name = str(s.name).lower()
        if any(t in name for t in ('id', 'code', 'key', 'index')):
            return s.is_unique
    return False

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = [c for c in train.columns if c not in drop and not key_like(train[c])]

vm = pd.read_csv(inputs['09_validate_score_validation_AE76P4WA/validation_metrics.csv'])
low = {c.lower(): c for c in vm.columns}
lower_better = str(primary).lower() in ('davies_bouldin', 'davies-bouldin', 'daviesbouldin')
if 'metric' in low and 'value' in low:
    mcol, vcol = low['metric'], low['value']
    model_col = low.get('model') or low.get('candidate') or vm.columns[0]
    sub = vm[vm[mcol].astype(str).str.lower() == str(primary).lower()].copy()
    if sub.empty:
        raise ValueError(f"primary metric {primary!r} not found; columns={list(vm.columns)}")
    sub['value'] = pd.to_numeric(sub[vcol], errors='coerce')
    sel = sub.dropna(subset=['value']).sort_values('value', ascending=lower_better)
    best_name = sel.iloc[0][model_col]
else:
    model_col = next((c for c in vm.columns if c.lower() in ('model', 'candidate', 'name')), vm.columns[0])
    pcol = primary if primary in vm.columns else next(c for c in vm.columns if c.lower() == str(primary).lower())
    vm[pcol] = pd.to_numeric(vm[pcol], errors='coerce')
    sel = vm.dropna(subset=[pcol]).sort_values(pcol, ascending=lower_better)
    best_name = sel.iloc[0][model_col]

s = str(best_name).lower()
if 'baseline' in s or 'sav13199' in s:
    model_path = inputs['07_model_train_baseline_SAV13199/model.pkl']
elif 'alternative' in s or 'sqa69r0g' in s:
    model_path = inputs['08_model_train_alternative_SQA69R0G/alternative_model.pkl']
elif 'latest' in s:
    model_path = inputs['latest_model']
else:
    raise ValueError(f"Cannot map validation model {best_name!r}; keys={list(inputs)}")

with open(model_path, 'rb') as f:
    model = pickle.load(f)
if isinstance(model, dict) and 'model' in model:
    model = model['model']

def input_cols(m, fallback):
    if hasattr(m, 'feature_names_in_'):
        return list(m.feature_names_in_)
    if hasattr(m, 'named_steps'):
        for step in m.named_steps.values():
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return fallback

cols = input_cols(model, feature_cols)
for df in (train, test):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing model input columns {missing}")

def embedding(m, X):
    if isinstance(m, Pipeline) and len(m.steps) > 1:
        return m[:-1].transform(X)
    if hasattr(m, 'transform'):
        return m.transform(X)
    return X

Xtr, Xte = train[cols], test[cols]
ytr = np.asarray(model.predict(Xtr)).ravel()
yte = np.asarray(model.predict(Xte)).ravel()

Ete = embedding(model, Xte)
if hasattr(Ete, 'toarray') and len(test) <= 20000:
    Ete = Ete.toarray()

n, k = len(Xte), len(np.unique(yte))
if n >= 2 and k >= 2:
    sil = float(silhouette_score(Ete, yte, sample_size=min(10000, n), random_state=42))
    ch = float(calinski_harabasz_score(Ete, yte))
    db = float(davies_bouldin_score(Ete, yte))
else:
    sil = ch = db = float('nan')

tm = pd.DataFrame({
    'model': [best_name] * 3,
    'metric': [primary, 'calinski_harabasz', 'davies_bouldin'],
    'value': [sil, ch, db],
})
mlca_runtime.save_table(tm, 'test_metrics.csv', caption='Test clustering metrics for frozen validation-selected candidate')
for name, value in tm[['metric', 'value']].itertuples(index=False):
    mlca_runtime.record_metric(str(name), float(value), 'test')
mlca_runtime.record_fact_candidate('test_score', float(sil), source='test_metrics.csv')

assign = pd.concat([
    pd.DataFrame({'row_id': train.index, 'partition': 'train', 'cluster': ytr}),
    pd.DataFrame({'row_id': test.index, 'partition': 'test', 'cluster': yte}),
], ignore_index=True)
mlca_runtime.save_row_outputs(assign, 'assignments.parquet')

print(f"selected={best_name}")
print(f"test_rows={len(test)} clusters={k}")
print(f"{primary}={sil:.6g}")
print("saved test_metrics.csv and assignments.parquet")
