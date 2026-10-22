import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column', None)
task = protocol['task']
primary_metric = task['primary_metric']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
print(f"train rows: {len(train)}, val rows: {len(val)}")

def is_key_like(col):
    s = train[col]
    if s.nunique() != len(s):
        return False
    if s.dtype == 'object' or pd.api.types.is_string_dtype(s.dtype):
        return True
    if pd.api.types.is_integer_dtype(s.dtype):
        lname = col.lower()
        if any(k in lname for k in ('id', 'key', 'index')):
            return True
    return False

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
feature_cols = [c for c in feature_cols if not is_key_like(c)]
print(f"feature columns: {len(feature_cols)}")

with open(inputs['05_preprocess_fit_preprocessing_6GF5BX9F/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_select_features_1B1C3SBH/selector.pkl'], 'rb') as f:
    selector = pickle.load(f)
with open(inputs['07_model_train_logistic_2M248THN/model.pkl'], 'rb') as f:
    model = pickle.load(f)

if isinstance(model, Pipeline):
    steps = list(model.named_steps.keys())
    if any(('preprocess' in s.lower() or 'select' in s.lower()) for s in steps):
        final_pipeline = model
    else:
        final_pipeline = Pipeline([('preprocessor', preprocessor), ('selector', selector), ('model', model)])
else:
    final_pipeline = Pipeline([('preprocessor', preprocessor), ('selector', selector), ('model', model)])

X_val = val[feature_cols].copy()
y_val = val[targets[0]].copy()

final_est = final_pipeline.named_steps[list(final_pipeline.named_steps.keys())[-1]] if isinstance(final_pipeline, Pipeline) else final_pipeline
if hasattr(final_est, 'classes_') and len(final_est.classes_) == 2:
    classes = final_est.classes_
    y_val_num = y_val.map({classes[0]: 0, classes[1]: 1})
else:
    y_val_num = y_val

result = permutation_importance(
    final_pipeline, X_val, y_val_num, scoring='roc_auc',
    n_repeats=20, random_state=42, n_jobs=-1
)

imp_df = pd.DataFrame({
    'feature': feature_cols,
    'importance_mean': result.importances_mean,
    'importance_sd': result.importances_std
}).sort_values('importance_mean', ascending=False).reset_index(drop=True)
top15 = imp_df.head(15).copy()
top15['importance_mean'] = top15['importance_mean'].round(4)
top15['importance_sd'] = top15['importance_sd'].round(4)
mlca_runtime.save_table(top15, 'interpretation.csv', caption='Permutation importance top 15 features')
top_feature = top15.iloc[0]['feature']
mlca_runtime.record_fact_candidate('top_feature', top_feature, source='interpretation.csv')
print(f"top feature: {top_feature}")
