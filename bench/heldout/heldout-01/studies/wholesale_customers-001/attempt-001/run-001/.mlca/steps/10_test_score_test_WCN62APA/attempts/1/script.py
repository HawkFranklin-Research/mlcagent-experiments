import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
task = protocol['task']; primary = task['primary_metric']
targets = protocol.get('targets') or []
excluded = protocol.get('excluded_columns') or []
group_col = protocol.get('group_column')

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
test = pd.read_parquet(inp['test'])

drop = set(targets) | set(excluded)
if group_col: drop.add(group_col)

def is_keylike(s):
    if s.nunique(dropna=False) != len(s): return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s): return True
    if pd.api.types.is_integer_dtype(s):
        vals = s.dropna().sort_values().to_numpy()
        return len(vals) == len(s) and (vals == np.arange(vals[0], vals[0]+len(vals))).all()
    return False

feature_cols = [c for c in train.columns if c not in drop and not is_keylike(train[c])]

vm = pd.read_csv(inp['09_validate_score_validation_QM40WVM3/validation_metrics.csv'])
if 'split' in vm.columns: vm = vm[vm['split'] == 'val']
mcol = next(c for c in ['model','candidate','name'] if c in vm.columns)
vcol = next((c for c in ['value','score','mean'] if c in vm.columns), None)
if vcol is None: vcol = [c for c in vm.columns if vm[c].dtype.kind in 'f' and c not in (mcol,'metric','split')][0]

vp = vm[vm['metric'] == primary]
if len(vp) == 0: vp = vm[vm['metric'].str.replace('-','_') == primary.replace('-','_')]
lower = primary.lower() in ('davies_bouldin','davies-bouldin')
best_row = vp.loc[vp[vcol].idxmin() if lower else vp[vcol].idxmax()]
best_name = best_row[mcol]

cand = {'baseline': inp['07_model_train_baseline_0H9X73N5/model.pkl'],
        'alternative': inp['08_model_train_alternative_6TXXHVGF/alternative_model.pkl']}
mpath = next((p for k, p in cand.items() if k in str(best_name).lower()), None)
if mpath is None:
    raise ValueError(f"Unknown candidate {best_name}; available {list(cand)}")

model = pickle.loads(mpath.read_bytes())
use_cols = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else feature_cols

Xt_df = test[use_cols]
labels_test = model.predict(Xt_df)
if hasattr(model, 'steps') and len(model.steps) > 1:
    Xt = np.asarray(model[:-1].transform(Xt_df))
else:
    Xt = np.asarray(Xt_df)
if hasattr(Xt, 'toarray'): Xt = Xt.toarray()
if Xt.ndim == 1: Xt = Xt.reshape(-1, 1)

n = len(Xt); ss = 10000 if n > 10000 else None
sil = float(silhouette_score(Xt, labels_test, sample_size=ss, random_state=42))
ch = float(calinski_harabasz_score(Xt, labels_test))
db = float(davies_bouldin_score(Xt, labels_test))

mdf = pd.DataFrame([
    {'model': best_name, 'metric': 'silhouette', 'value': float(f"{sil:.4g}")},
    {'model': best_name, 'metric': 'calinski_harabasz', 'value': float(f"{ch:.4g}")},
    {'model': best_name, 'metric': 'davies_bouldin', 'value': float(f"{db:.4g}")},
])
mlca_runtime.save_table(mdf, 'test_metrics.csv', 'Clustering metrics (test)')

match = mdf[mdf['metric'] == primary]
if len(match) == 0: match = mdf[mdf['metric'].str.replace('-','_') == primary.replace('-','_')]
primary_val = float(match.iloc[0]['value'])
mlca_runtime.record_metric(primary, primary_val, 'test')
mlca_runtime.record_fact_candidate('test_score', primary_val, source='test_metrics.csv')

labels_train = model.predict(train[use_cols])
assign = pd.concat([
    pd.DataFrame({'row_id': train.index.values, 'partition': 'train', 'cluster': labels_train}),
    pd.DataFrame({'row_id': test.index.values, 'partition': 'test', 'cluster': labels_test}),
], ignore_index=True)
mlca_runtime.save_row_outputs(assign, 'assignments.parquet')

print(f"train={len(train)} test={len(test)} feats={len(use_cols)}")
print(f"selected={best_name} clusters={len(np.unique(labels_test))}")
print(f"{primary}={primary_val:.4f}")
