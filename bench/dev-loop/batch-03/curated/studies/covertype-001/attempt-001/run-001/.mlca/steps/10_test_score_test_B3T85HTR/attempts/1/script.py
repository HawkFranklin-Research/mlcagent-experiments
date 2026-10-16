import mlca_runtime
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')
task = protocol['task']
primary_metric = task['primary_metric']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
test = pd.read_parquet(inputs['test'])

val_manifest = json.loads(inputs['09_validate_score_validation_VH3QP507/manifest.json'].read_text())
selected_key = None
for k in ['selected_model', 'best_model', 'model_path', 'selected_model_path', 'frozen_model', 'model']:
    if k in val_manifest:
        selected_key = k
        break
if selected_key is None:
    raise KeyError(f"Selected model key not found. Available keys: {list(val_manifest.keys())}")
selected_path = val_manifest[selected_key]
if isinstance(selected_path, dict):
    for k in ['path', 'model_path', 'file']:
        if k in selected_path:
            selected_path = selected_path[k]
            break

if selected_path in inputs:
    model_path = inputs[selected_path]
else:
    model_path = None
    for k, v in inputs.items():
        if v.name == selected_path or k.endswith(selected_path):
            model_path = v
            break
    if model_path is None:
        raise KeyError(f"Selected model path {selected_path} not found. Available keys: {list(inputs.keys())}")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

preprocessor = None
selector = None
if not isinstance(model, Pipeline) or 'preprocessor' not in model.named_steps:
    with open(inputs['05_preprocess_fit_preprocessing_1A6AJYGR/preprocessor.pkl'], 'rb') as f:
        preprocessor = pickle.load(f)
if not isinstance(model, Pipeline) or 'selector' not in model.named_steps:
    with open(inputs['06_features_select_features_JT24GMYZ/selector.pkl'], 'rb') as f:
        selector = pickle.load(f)

if isinstance(model, Pipeline):
    steps = list(model.named_steps.items())
    step_names = [name for name, _ in steps]
    if 'preprocessor' not in step_names:
        steps.insert(0, ('preprocessor', preprocessor))
    if 'selector' not in step_names:
        idx = step_names.index('preprocessor') + 1 if 'preprocessor' in step_names else 0
        steps.insert(idx, ('selector', selector))
    pipeline = Pipeline(steps)
else:
    pipeline = Pipeline([('preprocessor', preprocessor), ('selector', selector), ('model', model)])

feature_cols = None
for step in pipeline.named_steps.values():
    if hasattr(step, 'feature_names_in_'):
        feature_cols = list(step.feature_names_in_)
        break
if feature_cols is None:
    feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]

X_test = test[feature_cols]
y_test = test[targets[0]].values

proba = pipeline.predict_proba(X_test)
preds = pipeline.predict(X_test)
if 'model' in pipeline.named_steps:
    final_estimator = pipeline.named_steps['model']
else:
    final_estimator = pipeline.named_steps[list(pipeline.named_steps.keys())[-1]]
classes = final_estimator.classes_

roc_auc_full = roc_auc_score(y_test, proba, multi_class='ovr', average='macro', labels=classes)
f1_full = f1_score(y_test, preds, average='macro')

rng = np.random.default_rng(42)
n = len(y_test)
roc_aucs = []
f1s = []
for _ in range(1000):
    idx = rng.integers(0, n, n)
    y_b = y_test[idx]
    proba_b = proba[idx]
    preds_b = preds[idx]
    if len(np.unique(y_b)) < len(classes):
        continue
    try:
        roc = roc_auc_score(y_b, proba_b, multi_class='ovr', average='macro', labels=classes)
        f1 = f1_score(y_b, preds_b, average='macro')
        roc_aucs.append(roc)
        f1s.append(f1)
    except ValueError:
        continue

roc_ci = np.percentile(roc_aucs, [2.5, 97.5]) if roc_aucs else (np.nan, np.nan)
f1_ci = np.percentile(f1s, [2.5, 97.5]) if f1s else (np.nan, np.nan)

metrics_df = pd.DataFrame({
    'metric': ['roc_auc_ovr', 'f1_macro'],
    'value': [roc_auc_full, f1_full],
    'ci_low': [roc_ci[0], f1_ci[0]],
    'ci_high': [roc_ci[1], f1_ci[1]]
}).round(4)
mlca_runtime.save_table(metrics_df, 'test_metrics.csv', caption='Test metrics with 95% bootstrap CI')

mlca_runtime.record_metric('roc_auc_ovr', roc_auc_full, 'test', ci=(roc_ci[0], roc_ci[1]))
mlca_runtime.record_metric('f1_macro', f1_full, 'test', ci=(f1_ci[0], f1_ci[1]))
mlca_runtime.record_fact_candidate('test_roc_auc', roc_auc_full, source='test_metrics.csv')

print(f"Test rows: {len(test)}")
print(f"Features: {len(feature_cols)}")
print(f"roc_auc_ovr: {roc_auc_full:.4f} CI [{roc_ci[0]:.4f}, {roc_ci[1]:.4f}]")
print(f"f1_macro: {f1_full:.4f} CI [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")
