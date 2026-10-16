import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
for key in ['targets', 'excluded_columns', 'group_column', 'task']:
    if key not in protocol:
        raise KeyError(f"Missing key {key} in _protocol. Available top-level keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']
primary_metric = protocol['task']['primary_metric']

train = pd.read_parquet(inputs['train'])
test = pd.read_parquet(inputs['test'])

drop_cols = set(targets) | set(excluded) | {group_col}
candidate_cols = [c for c in train.columns if c not in drop_cols]
feature_cols = []
for c in candidate_cols:
    if train[c].dtype == 'object' or train[c].dtype.name == 'string':
        if train[c].nunique() == len(train):
            continue
    elif pd.api.types.is_integer_dtype(train[c]):
        if train[c].nunique() == len(train):
            continue
    feature_cols.append(c)

val_metrics = pd.read_csv(inputs['09_validate_score_validation_JM44JZJV/validation_metrics.csv'])
val_primary = val_metrics[val_metrics['metric'] == primary_metric]
if val_primary.empty:
    raise ValueError(f"No validation metrics for primary_metric {primary_metric}")
best_row = val_primary.loc[val_primary['value'].idxmax()]
best_model_name = best_row['model']

if 'baseline' in best_model_name.lower():
    model_path = inputs['07_model_train_baseline_DWY924K5/model.pkl']
elif 'alternative' in best_model_name.lower():
    model_path = inputs['08_model_train_alternative_3C755C37/alternative_model.pkl']
else:
    model_path = None
    for k in inputs:
        if best_model_name in k:
            model_path = inputs[k]
            break
    if model_path is None:
        raise ValueError(f"Cannot map model name {best_model_name} to a file.")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

if not isinstance(model, Pipeline):
    with open(inputs['05_preprocess_fit_preprocessing_S3SDNFRS/preprocessor.pkl'], 'rb') as f:
        preprocessor = pickle.load(f)
    with open(inputs['06_features_retain_features_RHW0T9SM/selector_all.pkl'], 'rb') as f:
        selector = pickle.load(f)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', model)
    ])

X_train = train[feature_cols]
X_test = test[feature_cols]
y_train = model.predict(X_train)
y_test = model.predict(X_test)

if isinstance(model, Pipeline):
    X_test_transformed = model[:-1].transform(X_test)
else:
    X_test_transformed = X_test

sample_size = min(10000, len(X_test_transformed))
sil = silhouette_score(X_test_transformed, y_test, sample_size=sample_size, random_state=42)
ch = calinski_harabasz_score(X_test_transformed, y_test)
db = davies_bouldin_score(X_test_transformed, y_test)

metrics_df = pd.DataFrame({
    'model': [best_model_name] * 3,
    'metric': ['silhouette', 'calinski_harabasz', 'davies_bouldin'],
    'value': [sil, ch, db]
})
metrics_df['value'] = metrics_df['value'].apply(lambda x: float(f'{x:.4g}'))
mlca_runtime.save_table(metrics_df, 'test_metrics.csv', caption='Test metrics for frozen candidate')

metric_map = {'silhouette': sil, 'calinski_harabasz': ch, 'davies_bouldin': db}
if primary_metric not in metric_map:
    raise ValueError(f"Unsupported primary_metric: {primary_metric}")
primary_value = metric_map[primary_metric]
mlca_runtime.record_metric(primary_metric, float(primary_value), split='test')
mlca_runtime.record_fact_candidate('test_score', float(primary_value), source='test_metrics.csv')

train_assign = pd.DataFrame({
    'row_id': train.index,
    'partition': 'train',
    'cluster': y_train
})
test_assign = pd.DataFrame({
    'row_id': test.index,
    'partition': 'test',
    'cluster': y_test
})
assignments = pd.concat([train_assign, test_assign], ignore_index=True)
mlca_runtime.save_row_outputs(assignments, 'assignments.parquet')

print(f"Selected model: {best_model_name}")
print(f"Test silhouette: {sil:.4f}")
print(f"Test CH: {ch:.4f}")
print(f"Test DB: {db:.4f}")
print(f"Assignments saved: {len(assignments)} rows")
