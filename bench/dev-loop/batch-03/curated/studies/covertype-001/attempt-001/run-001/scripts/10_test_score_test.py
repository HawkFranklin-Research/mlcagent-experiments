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

test = pd.read_parquet(inputs['test'])

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

model = load_pkl(inputs['latest_model'])
if isinstance(model, Pipeline):
    pipeline = model
else:
    pre = load_pkl(inputs['05_preprocess_fit_preprocessing_1A6AJYGR/preprocessor.pkl'])
    sel = load_pkl(inputs['06_features_select_features_JT24GMYZ/selector.pkl'])
    steps = []
    for name, obj in [('preprocessor', pre), ('selector', sel), ('model', model)]:
        if isinstance(obj, Pipeline):
            steps.extend(obj.steps)
        else:
            steps.append((name, obj))
    pipeline = Pipeline(steps)

feature_cols = None
if hasattr(pipeline, 'feature_names_in_'):
    feature_cols = list(pipeline.feature_names_in_)
else:
    for step in pipeline.named_steps.values():
        if hasattr(step, 'feature_names_in_'):
            feature_cols = list(step.feature_names_in_)
            break
if feature_cols is None:
    feature_cols = [c for c in test.columns if c not in targets and c not in excluded and c != group]

X_test = test[feature_cols]
y_test = test[targets[0]].values

proba = pipeline.predict_proba(X_test)
preds = pipeline.predict(X_test)

if hasattr(pipeline, 'classes_'):
    classes = pipeline.classes_
else:
    final = pipeline.named_steps[list(pipeline.named_steps.keys())[-1]]
    classes = final.classes_

roc_auc_full = roc_auc_score(y_test, proba, multi_class='ovr', average='macro', labels=classes)
f1_full = f1_score(y_test, preds, average='macro', labels=classes)

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
        f1 = f1_score(y_b, preds_b, average='macro', labels=classes)
        roc_aucs.append(roc)
        f1s.append(f1)
    except ValueError:
        continue

roc_ci = np.percentile(roc_aucs, [2.5, 97.5]) if roc_aucs else (np.nan, np.nan)
f1_ci = np.percentile(f1s, [2.5, 97.5]) if f1s else (np.nan, np.nan)

metrics_df = pd.DataFrame({
    'metric': [primary_metric, 'f1_macro'],
    'value': [roc_auc_full, f1_full],
    'ci_low': [roc_ci[0], f1_ci[0]],
    'ci_high': [roc_ci[1], f1_ci[1]]
}).round(4)
mlca_runtime.save_table(metrics_df, 'test_metrics.csv', caption='Test metrics with 95% bootstrap CI')

mlca_runtime.record_metric(primary_metric, roc_auc_full, 'test', ci=(roc_ci[0], roc_ci[1]))
mlca_runtime.record_metric('f1_macro', f1_full, 'test', ci=(f1_ci[0], f1_ci[1]))
mlca_runtime.record_fact_candidate('test_roc_auc', roc_auc_full, source='test_metrics.csv')

print(f"Test rows: {len(test)}")
print(f"Features: {len(feature_cols)}")
print(f"{primary_metric}: {roc_auc_full:.4f} CI [{roc_ci[0]:.4f}, {roc_ci[1]:.4f}]")
print(f"f1_macro: {f1_full:.4f} CI [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")
