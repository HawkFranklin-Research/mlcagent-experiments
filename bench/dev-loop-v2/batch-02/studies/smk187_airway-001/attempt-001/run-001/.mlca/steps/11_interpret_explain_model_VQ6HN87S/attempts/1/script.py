import mlca_runtime, json, pickle, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
warnings.filterwarnings('ignore')
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
primary = protocol['task']['primary_metric']
targets = protocol['targets']
excluded = protocol['excluded_columns']
ycol = targets[0]

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
print(f"rows train={len(train)} val={len(val)} metric={primary}")

model_path = inp['latest_model'] if 'latest_model' in inp else inp['07_model_train_logistic_KVA6214M/model.pkl']
with open(model_path, 'rb') as f:
    model = pickle.load(f)

if hasattr(model, 'feature_names_in_'):
    feats = list(model.feature_names_in_)
else:
    feats = [c for c in train.columns if c not in targets and c not in set(excluded)]
n_feat = len(feats)

Xval = val[feats].copy()
yval = val[ycol].values

def proba(df):
    return model.predict_proba(df[feats])[:, 1]

base = roc_auc_score(yval, proba(Xval))
print(f"n_features={n_feat} base_roc_auc={base:.4f}")

steps = list(model.steps) if hasattr(model, 'steps') else []
final = steps[-1][1] if steps else model
coef = getattr(final, 'coef_', None)
if coef is None:
    coef = getattr(final, 'feature_importances_', None)
coef = np.abs(np.asarray(coef).ravel()) if coef is not None else None

cur_names = list(feats)
if coef is not None:
    for name, step in steps[:-1]:
        if hasattr(step, 'get_support'):
            mask = step.get_support()
            if len(mask) == len(cur_names):
                cur_names = [cur_names[i] for i in np.where(mask)[0]]
        elif hasattr(step, 'get_feature_names_out'):
            try:
                out = list(step.get_feature_names_out(cur_names))
                if len(out) == len(cur_names):
                    cur_names = out
            except Exception:
                pass

if coef is not None and len(cur_names) == len(coef):
    coef_df = pd.DataFrame({'feature': cur_names, 'abs_coef': coef})
elif coef is not None:
    coef_df = pd.DataFrame({'feature': feats[:len(coef)], 'abs_coef': coef})
else:
    coef_df = pd.DataFrame({'feature': feats, 'abs_coef': 0.0})

if n_feat <= 500:
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, Xval, yval, scoring='roc_auc',
                               n_repeats=20, random_state=42, n_jobs=-1)
    df = pd.DataFrame({'feature': feats, 'importance_mean': r.importances_mean,
                       'importance_sd': r.importances_std})
    record = {'method': 'permutation_importance_full_schema',
              'n_permuted': n_feat, 'n_repeats': 20, 'scoring': 'roc_auc'}
else:
    k = min(50, len(coef_df))
    top = coef_df.sort_values('abs_coef', ascending=False).head(k)
    n_repeats = 5
    rng = np.random.RandomState(42)
    Xp = Xval.copy()
    rows = []
    for f in top['feature']:
        orig = Xp[f].values.copy()
        drops = []
        for _ in range(n_repeats):
            Xp[f] = rng.permutation(orig)
            try:
                s = roc_auc_score(yval, proba(Xp))
            except Exception:
                s = base
            drops.append(base - s)
        Xp[f] = orig
        rows.append({'feature': f, 'importance_mean': float(np.mean(drops)),
                     'importance_sd': float(np.std(drops))})
    df = pd.DataFrame(rows)
    record = {'method': 'coefficient_ranked_topK_permutation',
              'n_permuted': k, 'n_repeats': n_repeats, 'scoring': 'roc_auc'}

df = df.sort_values('importance_mean', ascending=False).head(15).reset_index(drop=True)
df['importance_mean'] = df['importance_mean'].round(6)
df['importance_sd'] = df['importance_sd'].round(6)
mlca_runtime.save_table(df, 'interpretation.csv',
                        caption='Top 15 frozen-model feature importances (permutation on val).')
mlca_runtime.save_json(record, 'interpretation_method.json', kind='report')

top_feature = str(df.iloc[0]['feature'])
mlca_runtime.record_fact_candidate('top_feature', top_feature, source='interpretation.csv')
mlca_runtime.record_metric('baseline_val_roc_auc', float(base), 'val')
mlca_runtime.record_metric('n_permuted_features', float(record['n_permuted']), 'val')
mlca_runtime.record_metric('permutation_repeats', float(record['n_repeats']), 'val')
print(f"top_feature={top_feature} mean_drop={df.iloc[0]['importance_mean']:.4f}")
