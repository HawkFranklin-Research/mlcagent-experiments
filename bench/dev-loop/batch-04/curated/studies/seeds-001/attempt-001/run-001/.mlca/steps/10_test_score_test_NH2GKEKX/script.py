import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol.get('targets', [])
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column', None)
primary_metric = protocol['task']['primary_metric']

train = pd.read_parquet(inputs['train'])
test = pd.read_parquet(inputs['test'])

drop_cols = set(targets) | set(excluded)
if group:
    drop_cols.add(group)
feature_cols = [c for c in train.columns if c not in drop_cols]
for c in list(feature_cols):
    if train[c].dtype.kind in 'OSU' and train[c].nunique() == len(train[c]) and train[c].notna().all():
        feature_cols.remove(c)
    elif pd.api.types.is_integer_dtype(train[c]) and train[c].nunique() == len(train[c]) and train[c].notna().all():
        if train[c].max() - train[c].min() == len(train[c]) - 1:
            feature_cols.remove(c)

val_m = pd.read_csv(inputs['09_validate_score_validation_B89RX1GQ/validation_metrics.csv'])
if 'metric' in val_m.columns and 'value' in val_m.columns:
    pv = val_m[val_m['metric'] == primary_metric].copy()
    if pv.empty:
        raise ValueError(f"No validation rows for {primary_metric}")
    higher_better = primary_metric in ('silhouette', 'calinski_harabasz')
    best_row = pv.loc[pv['value'].idxmax()] if higher_better else pv.loc[pv['value'].idxmin()]
    best_model = best_row['model']
else:
    if primary_metric not in val_m.columns:
        raise ValueError(f"Validation metrics missing column {primary_metric}")
    higher_better = primary_metric in ('silhouette', 'calinski_harabasz')
    best_row = val_m.loc[val_m[primary_metric].idxmax()] if higher_better else val_m.loc[val_m[primary_metric].idxmin()]
    best_model = best_row['model'] if 'model' in val_m.columns else best_row['candidate']

path_map = {}
for key, path in inputs.items():
    if key.endswith('model.pkl') or key == 'latest_model':
        path_map[key] = path
        fname = key.split('/')[-1].replace('.pkl', '')
        path_map[fname] = path
        if '/' in key:
            path_map[key.split('/')[0]] = path
path_map['baseline'] = inputs['07_model_train_baseline_BPBM7F9H/model.pkl']
path_map['alternative'] = inputs['08_model_train_alternative_09MZ25S1/alternative_model.pkl']
path_map['latest_model'] = inputs['latest_model']

model_path = None
for k, v in path_map.items():
    if best_model == k or best_model in k or k in best_model:
        model_path = v
        break
if model_path is None:
    raise ValueError(f"Cannot map {best_model} to a path. Available: {list(path_map.keys())}")

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(inputs['05_preprocess_fit_preprocessing_FVVN700K/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_retain_features_FW7XY320/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)

def make_pipeline(model, preprocessor, selector):
    if isinstance(model, Pipeline) and len(model.steps) > 1:
        return model
    steps = []
    if preprocessor is not None:
        steps.append(('preprocessor', preprocessor))
    if selector is not None:
        steps.append(('selector', selector))
    steps.append(('model', model))
    return Pipeline(steps=steps)

pipeline = make_pipeline(model, preprocessor, selector)

def get_features(pipeline, X):
    if isinstance(pipeline, Pipeline) and len(pipeline.steps) > 1:
        feature_pipe = Pipeline(pipeline.steps[:-1])
        return feature_pipe.transform(X)
    return X

X_test = test[feature_cols]
y_test = pipeline.predict(X_test)
X_test_trans = get_features(pipeline, X_test)

n = len(X_test_trans)
n_clusters = len(np.unique(y_test))
metrics = {'silhouette': np.nan, 'calinski_harabasz': np.nan, 'davies_bouldin': np.nan}
if n >= 2 and n_clusters >= 2:
    metrics['silhouette'] = float(silhouette_score(X_test_trans, y_test, sample_size=min(10000, n), random_state=42))
    metrics['calinski_harabasz'] = float(calinski_harabasz_score(X_test_trans, y_test))
    metrics['davies_bouldin'] = float(davies_bouldin_score(X_test_trans, y_test))

test_metrics = pd.DataFrame({
    'model': [best_model] * 3,
    'metric': ['silhouette', 'calinski_harabasz', 'davies_bouldin'],
    'value': [metrics['silhouette'], metrics['calinski_harabasz'], metrics['davies_bouldin']]
})
test_metrics['value'] = test_metrics['value'].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) else x)
mlca_runtime.save_table(test_metrics, 'test_metrics.csv', caption='Test metrics')

mlca_runtime.record_metric(primary_metric, metrics[primary_metric], split='test')
mlca_runtime.record_fact_candidate('test_score', metrics[primary_metric], source='test_metrics.csv')

X_train = train[feature_cols]
y_train = pipeline.predict(X_train)
train_assign = pd.DataFrame({'row_id': train.index, 'partition': 'train', 'cluster': y_train})
test_assign = pd.DataFrame({'row_id': test.index, 'partition': 'test', 'cluster': y_test})
assignments = pd.concat([train_assign, test_assign], ignore_index=True)
mlca_runtime.save_row_outputs(assignments, 'assignments.parquet')

print(f"Selected model: {best_model}")
print(f"Test {primary_metric}: {metrics[primary_metric]:.4f}")
print(f"Assignments rows: {len(assignments)}")
