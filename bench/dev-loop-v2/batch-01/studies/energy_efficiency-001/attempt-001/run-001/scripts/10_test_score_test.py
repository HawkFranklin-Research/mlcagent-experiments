import json, pickle, numpy as np, pandas as pd
import mlca_runtime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
targets = proto['targets']
primary = proto['task']['primary_metric']
excluded = proto['excluded_columns']
group = proto['group_column']


def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def predict_df(model, df):
    cols = getattr(model, 'feature_names_in_', None)
    if cols is not None:
        X = df[list(cols)]
    else:
        X = df[[c for c in df.columns if c not in targets and c not in excluded and c != group]]
    return np.asarray(model.predict(X))


test = pd.read_parquet(inp['test'])
val = pd.read_parquet(inp['val'])

cands = {
    'baseline': load(inp['07_model_train_baseline_AKKRPP8F/model.pkl']),
    'alternative': load(inp['08_model_train_alternative_HB44V41K/alternative_model.pkl']),
}


def macro_rmse(model, df):
    p = predict_df(model, df)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    return float(np.mean([np.sqrt(mean_squared_error(df[t].to_numpy(), p[:, i]))
                          for i, t in enumerate(targets)]))


val_scores = {name: macro_rmse(m, val) for name, m in cands.items()}
best_name = min(val_scores, key=val_scores.get)
best_model = cands[best_name]

p = predict_df(best_model, test)
if p.ndim == 1:
    p = p.reshape(-1, 1)

rows = []
for i, t in enumerate(targets):
    yt, yp = test[t].to_numpy(), p[:, i]
    rows.append({'model': f'best_{best_name}', 'metric': f'rmse__{t}',
                 'value': float(np.sqrt(mean_squared_error(yt, yp)))})
    rows.append({'model': f'best_{best_name}', 'metric': f'mae__{t}',
                 'value': float(mean_absolute_error(yt, yp))})
    rows.append({'model': f'best_{best_name}', 'metric': f'r2__{t}',
                 'value': float(r2_score(yt, yp))})

for mname, fn in [('rmse', lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp)))),
                  ('mae', lambda yt, yp: float(mean_absolute_error(yt, yp))),
                  ('r2', lambda yt, yp: float(r2_score(yt, yp)))]:
    vs = [fn(test[t].to_numpy(), p[:, i]) for i, t in enumerate(targets)]
    rows.append({'model': f'best_{best_name}', 'metric': mname, 'value': float(np.mean(vs))})

metrics_df = pd.DataFrame(rows)
metrics_df['value'] = metrics_df['value'].astype(float).round(6)
mlca_runtime.save_table(metrics_df, 'test_metrics.csv',
                        caption='Test metrics per target and unweighted macro average')

preds = pd.DataFrame({'row_id': test.index, 'partition': 'test'})
for i, t in enumerate(targets):
    preds[f'y_true__{t}'] = test[t].to_numpy()
    preds[f'y_pred__{t}'] = p[:, i]
mlca_runtime.save_row_outputs(preds, 'test_predictions.parquet')

test_score = float(metrics_df.loc[metrics_df['metric'] == primary, 'value'].iloc[0])
mlca_runtime.record_metric(primary, test_score, 'test')
mlca_runtime.record_fact_candidate('test_score', test_score, source='test_metrics.csv')

print(f"val macro rmse: { {k: round(v,4) for k,v in val_scores.items()} }")
print(f"chosen: {best_name}")
print(f"test {primary}: {test_score:.4f}")
