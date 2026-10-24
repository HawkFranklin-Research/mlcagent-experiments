import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
outdir = mlca_runtime.output_dir()

protocol = json.loads(inputs['_protocol'].read_text())
target = protocol['targets'][0]
task = protocol['task']
metric = task['primary_metric']
n_classes = task.get('n_classes', 2)

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
print(f"train rows: {len(train)}, val rows: {len(val)}")

with open(inputs['08_model_train_logistic_Z47S9AWS/model.pkl'], 'rb') as f:
    model = pickle.load(f)

feat_names = list(model.feature_names_in_)
n_feat = len(feat_names)
y_val = val[target].values
X_val = val[feat_names].copy()

if n_classes == 2:
    def scorer(est, X, y):
        p = est.predict_proba(X)[:, 1]
        return roc_auc_score(y, p)
else:
    def scorer(est, X, y):
        p = est.predict_proba(X)
        return roc_auc_score(y, p, multi_class='ovr', average='macro')

if n_feat <= 500:
    n_repeats = 20
    result = permutation_importance(model, X_val, y_val, scoring=scorer,
                                    n_repeats=n_repeats, random_state=42, n_jobs=-1)
    imp_mean = result.importances_mean
    imp_sd = result.importances_std
    method = f'permutation_importance on val (n_repeats={n_repeats}, n_features={n_feat})'
    order = np.argsort(imp_mean)[::-1]
    df_imp = pd.DataFrame({
        'feature': [feat_names[i] for i in order],
        'importance_mean': imp_mean[order],
        'importance_sd': imp_sd[order]
    })
else:
    coefs = None
    if hasattr(model, 'coef_'):
        c = np.abs(model.coef_).flatten()
        if len(c) == n_feat:
            coefs = c
    elif hasattr(model, 'feature_importances_'):
        c = model.feature_importances_
        if len(c) == n_feat:
            coefs = c
    if coefs is None and hasattr(model, 'named_steps'):
        last = list(model.named_steps.values())[-1]
        if hasattr(last, 'coef_'):
            c = np.abs(last.coef_).flatten()
            if len(c) == n_feat:
                coefs = c
        elif hasattr(last, 'feature_importances_'):
            c = last.feature_importances_
            if len(c) == n_feat:
                coefs = c
    if coefs is not None:
        n_repeats = 5
        top_k = min(50, n_feat)
        order = np.argsort(coefs)[::-1][:top_k]
        top_feats = [feat_names[i] for i in order]
        base_score = scorer(model, X_val, y_val)
        rng = np.random.RandomState(42)
        imp_dict = {}
        for f in top_feats:
            orig = X_val[f].values.copy()
            scores = []
            for _ in range(n_repeats):
                X_val[f] = rng.permutation(orig)
                scores.append(scorer(model, X_val, y_val))
            X_val[f] = orig
            scores = np.array(scores)
            imp_dict[f] = (base_score - scores.mean(), scores.std())
        rows = [{'feature': f, 'importance_mean': imp_dict[f][0],
                 'importance_sd': imp_dict[f][1]} for f in top_feats]
        df_imp = pd.DataFrame(rows).sort_values('importance_mean', ascending=False)
        method = f'coef/importance ranking then permutation on top {top_k} (n_repeats={n_repeats}, n_features={n_feat})'
    else:
        df_imp = pd.DataFrame(columns=['feature', 'importance_mean', 'importance_sd'])
        method = f'limited: no reliable feature importance for {n_feat} features'

df_imp = df_imp.head(50)
mlca_runtime.save_table(df_imp, 'interpretation.csv', caption='Feature importances')

if len(df_imp) > 0:
    top_feature = df_imp.iloc[0]['feature']
    mlca_runtime.record_fact_candidate('top_feature', top_feature, source='interpretation.csv')
    print(f"top feature: {top_feature}")

print(f"features: {n_feat}, method: {method}, saved rows: {len(df_imp)}")
