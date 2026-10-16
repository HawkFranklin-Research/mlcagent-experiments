import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

task = protocol['task']
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)
primary_metric = task.get('primary_metric') or protocol.get('primary_metric')
if not primary_metric:
    primary_metric = 'roc_auc_ovr' if task.get('type') == 'multiclass' else 'roc_auc'

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

def is_key_like(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.nunique() == len(s) and len(s) > 0
    if pd.api.types.is_integer_dtype(s):
        return s.nunique() == len(s) and len(s) > 0
    return False

feature_cols = [c for c in train.columns
                if c not in targets and c not in excluded
                and c != group_col and not is_key_like(train[c])]

def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

pre = load(inputs['05_preprocess_fit_preprocessing_Y4SGGFB6/preprocessor.pkl'])
sel = load(inputs['06_features_select_features_W4621TGT/selector.pkl'])
m1 = load(inputs['07_model_train_logistic_SNK2GEND/model.pkl'])
m2 = load(inputs['08_model_train_alternative_CWV9X77F/alternative_model.pkl'])

def build(est):
    if isinstance(est, Pipeline):
        return est
    if isinstance(sel, Pipeline):
        return Pipeline([('selector', sel), ('model', est)])
    return Pipeline([('preprocessor', pre), ('selector', sel), ('model', est)])

pipes = [('logistic', build(m1)), ('alternative', build(m2))]

def input_cols(pipe, fallback, val_cols):
    cols = None
    if hasattr(pipe, 'feature_names_in_'):
        cols = list(pipe.feature_names_in_)
    elif isinstance(pipe, Pipeline):
        first = pipe.steps[0][1]
        if hasattr(first, 'feature_names_in_'):
            cols = list(first.feature_names_in_)
    if cols and all(c in val_cols for c in cols):
        return cols
    return fallback

y_val = val[targets[0]]
rows = []
for name, pipe in pipes:
    cols = input_cols(pipe, feature_cols, val.columns)
    X = val[cols]
    proba = pipe.predict_proba(X)
    pred = pipe.predict(X)
    classes = pipe.classes_
    roc = roc_auc_score(y_val, proba, multi_class='ovr', average='macro', labels=classes)
    f1 = f1_score(y_val, pred, average='macro', labels=classes)
    rows.append({'model': name, 'metric': 'roc_auc_ovr', 'value': roc, 'threshold': np.nan})
    rows.append({'model': name, 'metric': 'f1_macro', 'value': f1, 'threshold': np.nan})
    val_primary = roc if primary_metric == 'roc_auc_ovr' else f1
    mlca_runtime.record_metric(primary_metric, float(val_primary), 'val')

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, 'validation_metrics.csv', caption='Validation metrics for candidate models')

sub = df[df['metric'] == primary_metric]
best = sub.loc[sub['value'].idxmax()]
best_model = best['model']
best_auc = df[(df['model'] == best_model) & (df['metric'] == 'roc_auc_ovr')].iloc[0]['value']
mlca_runtime.record_fact_candidate('best_model', best_model, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_roc_auc', float(best_auc), source='validation_metrics.csv')
print(f"val rows={len(val)} models={len(pipes)} best={best_model} auc={best_auc:.4f}")
