import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

for key in ['targets', 'excluded_columns', 'group_column', 'task']:
    if key not in protocol:
        raise KeyError(f"Missing '{key}' in protocol. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']
task = protocol['task']
primary_metric = task['primary_metric']
n_classes = task.get('n_classes', 2)

train_df = pd.read_parquet(inputs['train'])
val_df = pd.read_parquet(inputs['val'])

pipeline = None
for key in ['latest_model', '07_model_train_logistic_NCC1WZ5Q/model.pkl',
            '08_model_train_alternative_BY729WV6/alternative_model.pkl']:
    if key in inputs:
        with open(inputs[key], 'rb') as f:
            obj = pickle.load(f)
        if isinstance(obj, Pipeline) and hasattr(obj, 'predict_proba') and hasattr(obj, 'named_steps'):
            pipeline = obj
            break
if pipeline is None:
    raise ValueError("No complete pipeline found in the provided pickles.")

if hasattr(pipeline, 'feature_names_in_'):
    feature_cols = list(pipeline.feature_names_in_)
else:
    feature_cols = [c for c in train_df.columns
                    if c not in targets and c not in excluded
                    and (group_col is None or c != group_col)
                    and pd.api.types.is_numeric_dtype(train_df[c])]

feature_cols = [c for c in feature_cols
                if c not in targets and c not in excluded
                and (group_col is None or c != group_col)]

X_val = val_df[feature_cols].copy()
y_val = val_df[targets[0]]
classes = pipeline.classes_

if n_classes == 2:
    y_val_encoded = (y_val == classes[1]).astype(int)
else:
    y_val_encoded = y_val

def auc_scorer(estimator, X, y):
    proba = estimator.predict_proba(X)
    if n_classes == 2:
        return roc_auc_score(y, proba[:, 1])
    else:
        return roc_auc_score(y, proba, multi_class='ovr', labels=estimator.classes_)

result = permutation_importance(
    pipeline, X_val, y_val_encoded,
    scoring=auc_scorer,
    n_repeats=20,
    random_state=42,
    n_jobs=None
)

imp_df = pd.DataFrame({
    'feature': feature_cols,
    'importance_mean': result.importances_mean,
    'importance_sd': result.importances_std
})
imp_df = imp_df.sort_values('importance_mean', ascending=False).head(15).reset_index(drop=True)
for col in ['importance_mean', 'importance_sd']:
    imp_df[col] = imp_df[col].apply(lambda x: float(f"{x:.4g}"))

mlca_runtime.save_table(imp_df, "interpretation.csv", caption="Permutation importance on validation rows")

top_feature = imp_df.iloc[0]['feature']
mlca_runtime.record_fact_candidate('top_feature', top_feature, source='interpretation.csv')

print(f"Features used: {len(feature_cols)}")
print(f"Top feature: {top_feature}")
print(f"Mean importance: {imp_df.iloc[0]['importance_mean']}")
