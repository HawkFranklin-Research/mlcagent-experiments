import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance


def round_sig(x, sig=4):
    if not np.isfinite(x) or x == 0:
        return 0.0
    return float(round(x, -int(np.floor(np.log10(abs(x)))) + (sig - 1)))


mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol['targets']
excluded = protocol.get('excluded_columns') or []
group_col = protocol.get('group_column')
primary_metric = protocol['task'].get('primary_metric', 'roc_auc')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

def load_pkl(key):
    with open(inputs[key], 'rb') as f:
        return pickle.load(f)

pre = load_pkl('05_preprocess_fit_preprocessing_S343E7S5/preprocessor.pkl')
sel = load_pkl('06_features_select_features_K8ATBX97/selector.pkl')
clf = load_pkl('07_model_train_logistic_CT3F59V6/model.pkl')

def fn_in(obj):
    try:
        v = getattr(obj, 'feature_names_in_', None)
        return list(v) if v is not None else []
    except Exception:
        return []

clf_in, sel_in, pre_in = fn_in(clf), fn_in(sel), fn_in(pre)
forbidden = set(targets) | set(excluded)
if group_col:
    forbidden.add(group_col)

def ok(cols):
    return bool(cols) and all(c in val.columns for c in cols) and not (set(cols) & forbidden)

if ok(clf_in):
    pipeline, feature_cols = clf, clf_in
elif pre_in and sel_in and set(pre_in) == set(sel_in) and ok(sel_in):
    pipeline, feature_cols = Pipeline([('sel', sel), ('clf', clf)]), sel_in
elif ok(sel_in):
    pipeline, feature_cols = Pipeline([('sel', sel), ('clf', clf)]), sel_in
elif ok(pre_in):
    pipeline, feature_cols = Pipeline([('pre', pre), ('sel', sel), ('clf', clf)]), pre_in
else:
    feature_cols = [c for c in train.columns if c in val.columns and c not in forbidden]
    pipeline = Pipeline([('pre', pre), ('sel', sel), ('clf', clf)])

feature_cols = [c for c in feature_cols if c in val.columns and c not in forbidden]
X_val = val[feature_cols].copy()
y_val, _ = pd.factorize(val[targets[0]].values)

scoring = 'roc_auc' if primary_metric == 'roc_auc' else 'roc_auc_ovr'
result = permutation_importance(pipeline, X_val, y_val, n_repeats=20,
                                random_state=42, scoring=scoring, n_jobs=1)

imp_df = pd.DataFrame({
    'feature': feature_cols,
    'importance_mean': [round_sig(v) for v in result.importances_mean],
    'importance_sd': [round_sig(v) for v in result.importances_std],
}).sort_values('importance_mean', ascending=False).head(15).reset_index(drop=True)

mlca_runtime.save_table(imp_df, 'interpretation.csv',
                       caption='Permutation importance on validation rows')
top_feature = imp_df.iloc[0]['feature']
mlca_runtime.record_fact_candidate('top_feature', top_feature, source='interpretation.csv')

print(f"Val rows: {len(val)}, features: {len(feature_cols)}")
print(f"Top: {top_feature}")
