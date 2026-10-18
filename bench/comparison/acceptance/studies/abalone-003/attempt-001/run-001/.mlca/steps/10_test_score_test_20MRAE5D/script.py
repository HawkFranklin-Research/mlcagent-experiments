import mlca_runtime
import json, pickle, numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
task = protocol['task']
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)
primary_metric_raw = str(task['primary_metric'])
primary_metric = primary_metric_raw.lower()
minimize = primary_metric in {'rmse','mae','mse','mape','logloss','mean_squared_error','mean_absolute_error'}

best_model = None
try:
    val_manifest = json.loads(inputs['09_validate_score_validation_1WV4GQNX/manifest.json'].read_text())
except Exception:
    val_manifest = {}
for key in ['selected_model','best_model','winner','chosen_model','model_name']:
    v = val_manifest.get(key)
    if isinstance(v, str) and v:
        best_model = v
        break

val_metrics = pd.read_csv(inputs['09_validate_score_validation_1WV4GQNX/validation_metrics.csv'])
if best_model is None and 'metric' in val_metrics.columns and 'model' in val_metrics.columns:
    vm = val_metrics[val_metrics['metric'].astype(str).str.lower().str.startswith(primary_metric)].copy()
    scores = {}
    for m, g in vm.groupby('model'):
        macro = g[g['metric'].astype(str).str.lower().str.contains('macro', na=False)]
        scores[m] = float(macro['value'].iloc[0]) if not macro.empty else float(g['value'].mean())
    if scores:
        keys, vals = list(scores.keys()), list(scores.values())
        idx = int(np.argmin(vals)) if minimize else int(np.argmax(vals))
        best_model = keys[idx]

def pick_model_path(bm):
    s = str(bm).lower() if bm is not None else ''
    if 'baseline' in s:
        return inputs['07_model_train_baseline_3H582954/model.pkl']
    if 'alt' in s or 'latest' in s:
        return inputs['08_model_train_alternative_J64AD4FF/alternative_model.pkl']
    return inputs['latest_model']

with open(pick_model_path(best_model), 'rb') as f:
    model = pickle.load(f)

test_df = pd.read_parquet(inputs['test'])

def get_input_cols(obj):
    if hasattr(obj, 'feature_names_in_'):
        try:
            return list(obj.feature_names_in_)
        except Exception:
            pass
    if hasattr(obj, 'steps'):
        for _, step in obj.steps:
            cols = get_input_cols(step)
            if cols is not None:
                return cols
    return None

feature_cols = get_input_cols(model)
if feature_cols is None or not all(c in test_df.columns for c in feature_cols):
    drop = set(targets) | set(excluded)
    if group_col:
        drop.add(group_col)
    feature_cols = []
    for c in test_df.columns:
        if c in drop:
            continue
        s = test_df[c]
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique() == len(s) and len(s) > 5:
            continue
        feature_cols.append(c)

X_test = test_df[feature_cols]
y_test = np.asarray(test_df[targets], dtype=float)
y_pred = np.asarray(model.predict(X_test), dtype=float)
if y_pred.ndim == 1:
    y_pred = y_pred.reshape(-1, 1)
if y_test.ndim == 1:
    y_test = y_test.reshape(-1, 1)

rows = []
for i, tgt in enumerate(targets):
    yt, yp = y_test[:, i], y_pred[:, i]
    for name, val in [('rmse', float(np.sqrt(mean_squared_error(yt, yp)))),
                      ('mae', float(mean_absolute_error(yt, yp))),
                      ('r2', float(r2_score(yt, yp)))]:
        rows.append({'model': str(best_model), 'metric': f'{name}_{tgt}', 'value': val})
for name in ['rmse', 'mae', 'r2']:
    per = [r['value'] for r in rows if r['metric'].startswith(name + '_')]
    rows.append({'model': str(best_model), 'metric': f'{name}_macro', 'value': float(np.mean(per))})

test_metrics = pd.DataFrame(rows)
test_metrics['value'] = test_metrics['value'].astype(float).round(4)
mlca_runtime.save_table(test_metrics, "test_metrics.csv", caption="Test metrics for validation-selected model")

metric_lc = test_metrics['metric'].str.lower()
macro_row = test_metrics[metric_lc == f'{primary_metric}_macro']
if len(macro_row):
    primary_value = float(macro_row['value'].iloc[0])
else:
    primary_value = float(test_metrics[metric_lc.str.startswith(primary_metric)]['value'].mean())
mlca_runtime.record_metric(primary_metric_raw, primary_value, split='test')
mlca_runtime.record_fact_candidate('test_score', primary_value, source='test_metrics.csv')

print(f"Selected model: {best_model}")
print(f"Test rows: {len(test_df)}, features: {len(feature_cols)}")
print(f"{primary_metric} macro test: {primary_value:.4g}")
