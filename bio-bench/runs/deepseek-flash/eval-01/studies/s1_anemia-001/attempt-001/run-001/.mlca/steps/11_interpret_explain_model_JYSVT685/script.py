import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')
task = protocol['task']
primary_metric = task['primary_metric']
scoring = 'roc_auc' if primary_metric == 'roc_auc' else 'roc_auc_ovr'

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
print(f"train {train.shape}, val {val.shape}")

def load_pkl(key):
    with open(inputs[key], 'rb') as f:
        return pickle.load(f)

# Load saved objects
preproc = load_pkl('05_preprocess_fit_preprocessing_FMADE2YX/preprocessor.pkl')
selector = load_pkl('06_features_retain_features_SEM4Z3JD/selector_all.pkl')
model = load_pkl('07_model_train_logistic_0Q0G3DSJ/model.pkl')
alt = load_pkl('latest_model')

# Choose the most complete pipeline that accepts raw features
candidates = [
    ('latest_model', alt),
    ('07_model', model),
    ('08_model', load_pkl('08_model_train_alternative_EEA21607/alternative_model.pkl')),
]
chosen = None
for name, obj in candidates:
    if isinstance(obj, Pipeline) and hasattr(obj, 'predict_proba'):
        fni = getattr(obj, 'feature_names_in_', None)
        if fni is not None:
            # ensure target/excluded not in features
            bad = set(fni) & (set(targets) | set(excluded))
            if bad:
                raise ValueError(f"Target or excluded columns in features: {bad}")
            chosen = obj
            print(f"using pipeline {name} steps={list(obj.named_steps)}")
            break

if chosen is None:
    # Fallback: assemble using preproc + selector + model
    if isinstance(preproc, Pipeline) and getattr(preproc, 'feature_names_in_', None) is not None:
        if isinstance(selector, Pipeline) and getattr(selector, 'feature_names_in_', None) is not None:
            chosen = Pipeline([('selector', selector), ('model', model)])
        else:
            chosen = Pipeline([('preproc', preproc), ('selector', selector), ('model', model)])
        print(f"assembled pipeline steps={list(chosen.named_steps)}")
    else:
        raise ValueError("Could not find or assemble a pipeline that accepts raw features")

feature_names = list(chosen.feature_names_in_)
X_val = val[feature_names]
y_val = val[targets[0]].values

# Permutation importance
res = permutation_importance(
    chosen, X_val, y_val, scoring=scoring,
    n_repeats=20, random_state=42, n_jobs=1
)

imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': res.importances_mean,
    'importance_sd': res.importances_std,
}).sort_values('importance_mean', ascending=False).head(15).reset_index(drop=True)

imp_df['importance_mean'] = imp_df['importance_mean'].round(4)
imp_df['importance_sd'] = imp_df['importance_sd'].round(4)

mlca_runtime.save_table(imp_df, 'interpretation.csv', caption='Permutation importance (top 15)')
top_feature = imp_df.iloc[0]['feature']
mlca_runtime.record_fact_candidate('top_feature', top_feature, source='interpretation.csv')
print(f"top feature: {top_feature}")
