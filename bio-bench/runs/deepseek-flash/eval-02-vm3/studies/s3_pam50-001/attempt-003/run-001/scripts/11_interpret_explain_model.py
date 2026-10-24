import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
target = proto['targets'][0]
excluded = list(proto['excluded_columns'])
group = proto.get('group_column', None)
metric = proto['task']['primary_metric']
seed = 42

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
print(f"train {train.shape[0]}x{train.shape[1]} val {val.shape[0]}x{val.shape[1]}")

with open(inp['latest_model'], 'rb') as f:
    model = pickle.load(f)

def is_keylike(s):
    n = len(s)
    if s.dtype == object:
        return s.nunique(dropna=True) == n
    if s.dtype.kind in 'iu':
        return s.nunique(dropna=True) == n
    return False

feat_cols = [c for c in train.columns if c not in proto['targets'] and c not in excluded and c != group]
feat_cols = [c for c in feat_cols if not is_keylike(train[c])]
Xcols = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else feat_cols
n_feat = len(Xcols)
X_val = val[Xcols].copy()
y_val = val[target].values

def extract_imp(model):
    if hasattr(model, 'feature_importances_'):
        a = np.asarray(model.feature_importances_, float)
        if len(a) == n_feat: return a
    if hasattr(model, 'coef_'):
        c = np.asarray(model.coef_)
        a = np.abs(c).mean(0) if c.ndim > 1 else np.abs(c)
        if len(a) == n_feat: return a.astype(float)
    if hasattr(model, 'steps'):
        est = model.steps[-1][1]
        masks = []
        for _, step in model.steps[:-1]:
            if hasattr(step, 'get_support'):
                try: masks.append(step.get_support())
                except Exception: pass
        if hasattr(est, 'coef_'):
            c = np.asarray(est.coef_)
            a = np.abs(c).mean(0) if c.ndim > 1 else np.abs(c)
        elif hasattr(est, 'feature_importances_'):
            a = np.asarray(est.feature_importances_, float)
        else:
            return None
        for m in masks:
            if len(m) == n_feat and len(a) == m.sum():
                full = np.full(n_feat, np.nan)
                full[m] = a
                return full
    return None

def scores(X):
    return model.predict_proba(X) if hasattr(model, 'predict_proba') else model.decision_function(X)

def score_proba(p):
    return roc_auc_score(y_val, p, multi_class='ovr', average='macro')

if n_feat <= 500:
    method = f"perm_val_r20_f{n_feat}"
    r = permutation_importance(model, X_val, y_val, scoring=metric,
                               n_repeats=20, random_state=seed, n_jobs=-1)
    imp_mean, imp_sd = r.importances_mean, r.importances_std
    budget = f"n_features={n_feat}, repeats=20"
else:
    imp_full = extract_imp(model)
    if imp_full is not None and np.any(~np.isnan(imp_full)):
        top_idx = np.argsort(-np.nan_to_num(imp_full, nan=-np.inf))[:50]
        rng = np.random.RandomState(seed)
        base = score_proba(scores(X_val))
        imp_mean = np.full(n_feat, np.nan)
        imp_sd = np.full(n_feat, np.nan)
        for i in top_idx:
            col = Xcols[i]
            deltas = []
            for _ in range(5):
                Xp = X_val.copy()
                Xp[col] = rng.permutation(Xp[col].values)
                deltas.append(base - score_proba(scores(Xp)))
            imp_mean[i] = float(np.mean(deltas))
            imp_sd[i] = float(np.std(deltas))
        method = f"perm_top50_r5_f{n_feat}"
        budget = f"n_features={n_feat}, permuted=50, repeats=5"
    else:
        method = "limitation_no_reliable_ranking"
        imp_mean = np.full(n_feat, np.nan)
        imp_sd = np.full(n_feat, np.nan)
        budget = f"n_features={n_feat}, repeats=0"
        print("No reliable importance ranking available")

df = pd.DataFrame({'feature': Xcols, 'importance_mean': imp_mean, 'importance_sd': imp_sd})
df = df.dropna(subset=['importance_mean']).sort_values('importance_mean', ascending=False).head(15).reset_index(drop=True)
df[['importance_mean', 'importance_sd']] = df[['importance_mean', 'importance_sd']].round(4)
mlca_runtime.save_table(df, 'interpretation.csv', caption=f"{method}; {budget}")
if len(df):
    mlca_runtime.record_fact_candidate('top_feature', str(df.iloc[0]['feature']), source='interpretation.csv')
print(f"method={method} top={df.iloc[0]['feature'] if len(df) else 'none'}")
