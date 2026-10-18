import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
task = proto['task']; primary = task['primary_metric']
targets = proto.get('targets', []) or []
excluded = proto.get('excluded_columns', []) or []
group = proto.get('group_column')
drop = set(targets) | set(excluded)
if group: drop.add(str(group))

train = pd.read_parquet(inp['train'])
test = pd.read_parquet(inp['test'])

vm = pd.read_csv(inp['09_validate_score_validation_SVSGG8S9/validation_metrics.csv'])
if 'model' not in vm.columns or 'value' not in vm.columns or 'metric' not in vm.columns:
    raise ValueError(f"validation_metrics.csv columns: {list(vm.columns)}")
sub = vm[vm['metric'] == primary].dropna(subset=['value'])
best_model = str(sub.loc[sub['value'].idxmax(), 'model'])

with open(inp['07_model_train_baseline_FB68KDQ5/model.pkl'], 'rb') as f: base = pickle.load(f)
with open(inp['08_model_train_alternative_A16CW8P2/alternative_model.pkl'], 'rb') as f: alt = pickle.load(f)
cands = {'baseline': base, 'alternative': alt}
model = cands.get(best_model, base)
if isinstance(model, Pipeline):
    for k, v in model.named_steps.items():
        if k != 'model' and k != 'cluster':
            pass

fn = getattr(model, 'feature_names_in_', None)
if fn is not None:
    feat_cols = [c for c in fn if c in train.columns and c in test.columns and c not in drop]
else:
    feat_cols = []
    for c in train.columns:
        if c in drop: continue
        s = train[c]
        if s.dtype.kind in 'iu' and s.is_unique and len(s) > 1: continue
        if s.dtype == object and s.is_unique and len(s) > 1: continue
        feat_cols.append(c)
    feat_cols = [c for c in feat_cols if c in test.columns]

Xtr = train[feat_cols]
Xte = test[feat_cols]
if not hasattr(model, 'predict'):
    raise RuntimeError(f"frozen candidate {best_model} has no predict")

labels_tr = np.asarray(model.predict(Xtr))
labels_te = np.asarray(model.predict(Xte))

if isinstance(model, Pipeline) and len(model.steps) > 1:
    feat_pipe = Pipeline(model.steps[:-1])
    if hasattr(feat_pipe, 'feature_names_in_'):
        Zte = feat_pipe.transform(Xte[list(feat_pipe.feature_names_in_)])
    else:
        Zte = feat_pipe.transform(Xte)
else:
    Zte = np.asarray(Xte)

n_te = len(Zte)
ss = min(10000, n_te)
sil = float(silhouette_score(Zte, labels_te, sample_size=ss, random_state=42))
ch = float(calinski_harabasz_score(Zte, labels_te))
db = float(davies_bouldin_score(Zte, labels_te))

tm = pd.DataFrame([
    {'model': best_model, 'metric': primary, 'value': round(sil, 6)},
    {'model': best_model, 'metric': 'calinski_harabasz', 'value': round(ch, 6)},
    {'model': best_model, 'metric': 'davies_bouldin', 'value': round(db, 6)},
])
mlca_runtime.save_table(tm, 'test_metrics.csv', caption='Test clustering metrics (validation-frozen candidate)')
mlca_runtime.record_metric(primary, sil, 'test')
mlca_runtime.record_fact_candidate('test_score', sil, source='test_metrics.csv')

asg = pd.concat([
    pd.DataFrame({'row_id': train.index, 'partition': 'train', 'cluster': labels_tr}),
    pd.DataFrame({'row_id': test.index, 'partition': 'test', 'cluster': labels_te}),
], ignore_index=True)
mlca_runtime.save_row_outputs(asg, 'assignments.parquet')

print(f"selected={best_model} metric={primary} k_test={len(np.unique(labels_te))} n_test={n_te}")
print(f"silhouette={sil:.4f} ch={ch:.2f} db={db:.4f}")
