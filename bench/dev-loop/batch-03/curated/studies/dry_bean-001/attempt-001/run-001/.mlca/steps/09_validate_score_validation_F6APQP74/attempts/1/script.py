import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

required = ['task', 'primary_metric', 'targets', 'excluded_columns', 'group_column']
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing keys: {missing}. Available: {list(protocol.keys())}")

task = protocol['task']
primary_metric = protocol['primary_metric']
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

def is_key_like(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.nunique() == len(s) and len(s) > 0
    if pd.api.types.is_integer_dtype(s):
        return s.nunique() == len(s) and len(s) > 0
    return False

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col and not is_key_like(train[c])]

with open(inputs['05_preprocess_fit_preprocessing_Y4SGGFB6/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_select_features_W4621TGT/selector.pkl'], 'rb') as f:
    selector = pickle.load(f)
with open(inputs['07_model_train_logistic_SNK2GEND/model.pkl'], 'rb') as f:
    model_log = pickle.load(f)
with open(inputs['08_model_train_alternative_CWV9X77F/alternative_model.pkl'], 'rb') as f:
    model_alt = pickle.load(f)

def build_pipeline(est, pre, sel):
    if isinstance(est, Pipeline):
        return est
    if isinstance(sel, Pipeline):
        return Pipeline([('selector', sel), ('model', est)])
    return Pipeline([('preprocessor', pre), ('selector', sel), ('model', est)])

pipe_log = build_pipeline(model_log, preprocessor, selector)
pipe_alt = build_pipeline(model_alt, preprocessor, selector)

def get_input_cols(pipe, fallback):
    if hasattr(pipe, 'feature_names_in_'):
        return list(pipe.feature_names_in_)
    if isinstance(pipe, Pipeline):
        first = pipe.steps[0][1]
        if hasattr(first, 'feature_names_in_'):
            return list(first.feature_names_in_)
    return fallback

cols_log = get_input_cols(pipe_log, feature_cols)
cols_alt = get_input_cols(pipe_alt, feature_cols)

for c in cols_log:
    if c not in val.columns:
        raise KeyError(f"Missing column {c} for logistic model")
for c in cols_alt:
    if c not in val.columns:
        raise KeyError(f"Missing column {c} for alternative model")

X_val_log = val[cols_log]
X_val_alt = val[cols_alt]
y_val = val[targets[0]]

results = []
models = [('logistic', pipe_log, X_val_log), ('alternative', pipe_alt, X_val_alt)]

for name, pipe, X in models:
    proba = pipe.predict_proba(X)
    pred = pipe.predict(X)
    classes = pipe.classes_
    roc = roc_auc_score(y_val, proba, multi_class='ovr', average='macro', labels=classes)
    f1 = f1_score(y_val, pred, average='macro')
    results.append((name, 'roc_auc_ovr', roc, np.nan))
    results.append((name, 'f1_macro', f1, np.nan))
    if primary_metric == 'roc_auc_ovr':
        mlca_runtime.record_metric('roc_auc_ovr', roc, 'val')
    elif primary_metric == 'f1_macro':
        mlca_runtime.record_metric('f1_macro', f1, 'val')

df_res = pd.DataFrame(results, columns=['model', 'metric', 'value', 'threshold'])
mlca_runtime.save_table(df_res, 'validation_metrics.csv', caption='Validation metrics for candidate models')

primary_rows = df_res[df_res['metric'] == primary_metric]
if len(primary_rows) == 0:
    raise ValueError(f"Primary metric {primary_metric} not found in results")
best_row = primary_rows.loc[primary_rows['value'].idxmax()]
best_model = best_row['model']
best_val_auc = df_res[(df_res['model'] == best_model) & (df_res['metric'] == 'roc_auc_ovr')].iloc[0]['value']

mlca_runtime.record_fact_candidate('best_model', best_model, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_roc_auc', float(best_val_auc), source='validation_metrics.csv')

print(f"Models evaluated: {len(models)}")
print(f"Validation rows: {len(val)}")
print(f"Best model: {best_model}")
print(f"Best roc_auc_ovr: {best_val_auc:.4f}")
