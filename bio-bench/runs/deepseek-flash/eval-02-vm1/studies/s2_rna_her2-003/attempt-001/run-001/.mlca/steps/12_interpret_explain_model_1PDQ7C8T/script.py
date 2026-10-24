import mlca_runtime
import pickle, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
targets = proto['targets']
excluded = list(proto['excluded_columns'])
group_col = proto.get('group_column')
task = proto['task']
n_classes = int(task.get('n_classes', 2))

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
with open(inp['08_model_train_logistic_YSEARS1W/model.pkl'], 'rb') as f:
    model = pickle.load(f)

if hasattr(model, 'feature_names_in_'):
    feats = [str(c) for c in model.feature_names_in_]
else:
    ex = set(excluded) | set(targets)
    if group_col:
        ex.add(group_col)
    feats = [c for c in train.columns if c not in ex]

target = targets[0]
Xval = val[feats].copy()
yval = val[target].values
if n_classes == 2:
    classes = getattr(model, 'classes_', None)
    if classes is None and hasattr(model, 'steps'):
        classes = getattr(model.steps[-1][1], 'classes_', None)
    pos = classes[1] if classes is not None and len(classes) == 2 else np.unique(yval)[-1]
    yval = (yval == pos).astype(int)

n_feats = len(feats)

def score_it(X, y):
    p = model.predict_proba(X)
    if n_classes == 2:
        return roc_auc_score(y, p[:, 1])
    return roc_auc_score(y, p, multi_class='ovr', average='macro')

if n_feats <= 500:
    n_rep = 20
    perm_feats = list(feats)
    method = f'permutation_all_{n_rep}repeats'
else:
    raw_imps = None
    if hasattr(model, 'steps'):
        final = model.steps[-1][1]
        coef = getattr(final, 'coef_', None)
        if coef is None:
            coef = getattr(final, 'feature_importances_', None)
        if coef is not None:
            coef = np.abs(np.atleast_2d(coef))
            coef_v = coef.mean(axis=0) if coef.shape[0] > 1 else coef.ravel()
            for _, step in model.steps[:-1]:
                if hasattr(step, 'get_support'):
                    sup = step.get_support()
                    if len(sup) == n_feats and int(sup.sum()) == len(coef_v):
                        expanded = np.zeros(n_feats); expanded[sup] = coef_v
                        raw_imps = pd.Series(expanded, index=feats); break
            if raw_imps is None and len(coef_v) == n_feats:
                raw_imps = pd.Series(coef_v, index=feats)
        if raw_imps is not None:
            raw_imps = raw_imps.sort_values(ascending=False)
    if raw_imps is None:
        df = pd.DataFrame({'feature': ['UNAVAILABLE'], 'importance_mean': [np.nan], 'importance_sd': [np.nan]})
        method = 'limitation:no_raw_level_importance'
        perm_feats = []
        n_rep = 0
    else:
        perm_feats = list(raw_imps.index[:50])
        n_rep = 5
        method = f'coef_rank_top50_permute_{n_rep}repeats'

if perm_feats:
    base = score_it(Xval, yval)
    rng = np.random.RandomState(42)
    m, s = [], []
    for f in perm_feats:
        orig = Xval[f].values.copy()
        d = []
        for _ in range(n_rep):
            Xval[f] = rng.permutation(orig)
            d.append(base - score_it(Xval, yval))
        Xval[f] = orig
        m.append(float(np.mean(d))); s.append(float(np.std(d)))
    df = pd.DataFrame({'feature': perm_feats, 'importance_mean': m, 'importance_sd': s})

df = df.sort_values('importance_mean', ascending=False, na_position='last').head(15).reset_index(drop=True)
mlca_runtime.save_table(df, 'interpretation.csv', caption=f'method={method}; features_permuted={len(perm_feats)}; repeats={n_rep}')
if pd.notna(df['importance_mean'].iloc[0]):
    mlca_runtime.record_fact_candidate('top_feature', str(df['feature'].iloc[0]), source='interpretation.csv')
    mlca_runtime.record_metric('top_feature_importance_mean', float(df['importance_mean'].iloc[0]), split='val')
print(f'val_rows={len(val)} features={n_feats} permuted={len(perm_feats)} repeats={n_rep}')
print(f'method={method}')
