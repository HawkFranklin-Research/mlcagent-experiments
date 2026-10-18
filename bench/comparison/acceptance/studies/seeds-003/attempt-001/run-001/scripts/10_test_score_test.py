import mlca_runtime, json, pickle
import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs['_protocol'].read_text())
primary = proto['task']['primary_metric']
targets = proto.get('targets', [])
excluded = proto.get('excluded_columns', [])
group = proto.get('group_column', None)
if primary is None:
    primary = 'silhouette'

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
test = pd.read_parquet(inputs['test'])

feat_cols = [c for c in train.columns if c not in list(targets) + list(excluded) and c != group]
feat_cols = [c for c in feat_cols if not (train[c].dtype == object and train[c].nunique() == len(train))]
feat_cols = [c for c in feat_cols if not (not np.issubdtype(train[c].dtype, np.floating) and train[c].nunique() == len(train))]

def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

pre = load(inputs['05_preprocess_fit_preprocessing_0JVWWBDZ/preprocessor.pkl'])
sel = load(inputs['06_features_retain_features_JQHA8ERZ/selector_all.pkl'])
model_b = load(inputs['07_model_train_baseline_7GET20JD/model.pkl'])
model_a = load(inputs['08_model_train_alternative_VX601NWG/alternative_model.pkl'])

def complete(obj):
    if isinstance(obj, Pipeline):
        names = set(obj.named_steps.keys())
        if names & {'preprocessor', 'pre', 'selector', 'sel', 'preprocess'} or hasattr(obj, 'feature_names_in_'):
            return obj
    return Pipeline([('preprocessor', pre), ('selector', sel), ('model', obj)])

pipe_b = complete(model_b)
pipe_a = complete(model_a)

def transform_X(pipe, X):
    if isinstance(pipe, Pipeline):
        return pipe[:-1].transform(X)
    return X

def val_sil(pipe):
    X = val[feat_cols]
    labels = pipe.predict(X)
    if len(set(labels)) < 2:
        return -1.0
    Xt = transform_X(pipe, X)
    return silhouette_score(Xt, labels, sample_size=min(10000, len(X)), random_state=42)

sb, sa = val_sil(pipe_b), val_sil(pipe_a)
frozen, frozen_name = (pipe_b, 'baseline') if sb >= sa else (pipe_a, 'alternative')

X_test = test[feat_cols]
y_test = frozen.predict(X_test)
Xt = transform_X(frozen, X_test)
ns = min(10000, len(X_test))
sil = float(silhouette_score(Xt, y_test, sample_size=ns, random_state=42))
ch = float(calinski_harabasz_score(Xt, y_test))
db = float(davies_bouldin_score(Xt, y_test))

metrics_df = pd.DataFrame([
    {'model': frozen_name, 'metric': primary, 'value': round(sil, 4)},
    {'model': frozen_name, 'metric': 'calinski_harabasz', 'value': round(ch, 4)},
    {'model': frozen_name, 'metric': 'davies_bouldin', 'value': round(db, 4)},
])
mlca_runtime.save_table(metrics_df, 'test_metrics.csv')

mlca_runtime.record_metric(primary, sil, split='test')
mlca_runtime.record_fact_candidate('test_score', sil, source='test_metrics.csv')

train_pred = frozen.predict(train[feat_cols])
assignments = pd.concat([
    pd.DataFrame({'row_id': train.index, 'partition': 'train', 'cluster': train_pred}),
    pd.DataFrame({'row_id': test.index, 'partition': 'test', 'cluster': y_test}),
], ignore_index=True)
mlca_runtime.save_row_outputs(assignments, 'assignments.parquet')

print(f"primary={primary} frozen={frozen_name}")
print(f"test {primary}={sil:.4f} ch={ch:.2f} db={db:.4f}")
print(f"assignments={assignments.shape} n_train={len(train)} n_test={len(test)}")
