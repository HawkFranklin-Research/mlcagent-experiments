import mlca_runtime, json, pickle
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
primary_metric = protocol['task']['primary_metric']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
test = pd.read_parquet(inputs['test'])

def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

candidates = {
    'baseline': load(inputs['07_model_train_baseline_FFKH7YPA/model.pkl']),
    'alternative': load(inputs['08_model_train_alternative_TEGYDRX5/alternative_model.pkl']),
}

def transform_space(model, X):
    if isinstance(model, Pipeline) and len(model.steps) > 1:
        return model[:-1].transform(X)
    return X

def fit_score(model, df):
    cols = list(model.feature_names_in_)
    X = df[cols]
    labels = model.predict(X)
    Xt = transform_space(model, X)
    n = min(10000, len(Xt))
    if len(set(labels)) < 2:
        return -1.0, labels, Xt
    return silhouette_score(Xt, labels, sample_size=n, random_state=42), labels, Xt

val_scores = {name: fit_score(m, val)[0] for name, m in candidates.items()}
winner_name = max(val_scores, key=val_scores.get)
winner = candidates[winner_name]

sil, labels_test, Xt_test = fit_score(winner, test)
ch = calinski_harabasz_score(Xt_test, labels_test)
db = davies_bouldin_score(Xt_test, labels_test)

metrics_df = pd.DataFrame({
    'model': [winner_name] * 3,
    'metric': ['silhouette', 'calinski_harabasz', 'davies_bouldin'],
    'value': [round(float(sil), 4), round(float(ch), 4), round(float(db), 4)],
})
mlca_runtime.save_table(metrics_df, 'test_metrics.csv',
                        caption='Test metrics for validation-frozen candidate.')

mlca_runtime.record_metric(primary_metric, float(sil), 'test')
mlca_runtime.record_fact_candidate('test_score', float(sil), source='test_metrics.csv')

def assign(model, df, part):
    X = df[list(model.feature_names_in_)]
    labels = model.predict(X)
    return pd.DataFrame({'row_id': df.index, 'partition': part, 'cluster': labels})

assignments = pd.concat(
    [assign(winner, train, 'train'), assign(winner, test, 'test')],
    ignore_index=True,
)
mlca_runtime.save_row_outputs(assignments, 'assignments.parquet')

print(f"winner={winner_name} val_silh={val_scores[winner_name]:.4f}")
print(f"test silhouette={sil:.4f} ch={ch:.2f} db={db:.4f}")
print(f"assignments rows={len(assignments)}")
