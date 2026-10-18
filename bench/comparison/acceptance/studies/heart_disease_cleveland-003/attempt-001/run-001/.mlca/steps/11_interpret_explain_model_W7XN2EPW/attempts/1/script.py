import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

SEED = 42
mlca_runtime.set_seed(SEED)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
task = protocol['task']
primary_metric = task['primary_metric']
group_col = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=False) == len(s):
            continue
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=False) == len(s):
            continue
    feature_cols.append(c)

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

best_pipe = None
best_steps = -1
for key in [
    'latest_model',
    '08_model_train_alternative_MAM5EC9Z/alternative_model.pkl',
    '07_model_train_logistic_H9QQPD1P/model.pkl',
    '06_features_select_features_R97WA2VA/selector.pkl',
    '05_preprocess_fit_preprocessing_S0BP8ZJM/preprocessor.pkl',
]:
    if key not in inputs:
        continue
    try:
        obj = load_pkl(inputs[key])
    except Exception:
        continue
    if not isinstance(obj, Pipeline) or not hasattr(obj, 'predict_proba'):
        continue
    try:
        Xt = val[feature_cols].head(2)
        if hasattr(obj, 'feature_names_in_'):
            fn = list(obj.feature_names_in_)
            if all(c in Xt.columns for c in fn):
                Xt = Xt[fn]
        obj.predict_proba(Xt)
    except Exception:
        continue
    steps = len(getattr(obj, 'named_steps', {}))
    if steps > best_steps:
        best_steps = steps
        best_pipe = obj

if best_pipe is None:
    raise RuntimeError('No complete frozen pipeline accepts the raw feature DataFrame')

if hasattr(best_pipe, 'feature_names_in_'):
    input_cols = list(best_pipe.feature_names_in_)
    missing = [c for c in input_cols if c not in feature_cols]
    if missing:
        raise ValueError(f'Pipeline expects unavailable feature columns: {missing}')
else:
    input_cols = list(feature_cols)

X_val = val[input_cols].copy()
y_val = val[targets[0]].copy()

scoring = 'roc_auc' if primary_metric in ('roc_auc', 'auc') else primary_metric
perm = permutation_importance(
    best_pipe, X_val, y_val,
    scoring=scoring, n_repeats=20, random_state=SEED, n_jobs=1,
)

imp = pd.DataFrame({
    'feature': input_cols,
    'importance_mean': perm.importances_mean,
    'importance_sd': perm.importances_std,
}).sort_values('importance_mean', ascending=False).head(15).reset_index(drop=True)

for col in ['importance_mean', 'importance_sd']:
    imp[col] = imp[col].map(lambda v: float(f'{v:.4g}'))

mlca_runtime.save_table(imp, 'interpretation.csv', caption='Permutation importance on validation rows')
top_feature = imp.iloc[0]['feature']
mlca_runtime.record_fact_candidate('top_feature', top_feature, source='interpretation.csv')

print(f"features={len(input_cols)} val_rows={len(X_val)}")
print(f"top_feature={top_feature} importance_mean={imp.iloc[0]['importance_mean']}")
print("saved interpretation.csv")
