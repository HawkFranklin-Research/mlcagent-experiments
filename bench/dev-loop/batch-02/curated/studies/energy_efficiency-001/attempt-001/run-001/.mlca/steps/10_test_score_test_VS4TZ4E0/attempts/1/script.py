import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

required = ['targets', 'excluded_columns', 'group_column', 'task']
missing = [k for k in required if k not in protocol]
if missing:
    raise ValueError(f"Missing protocol keys: {missing}. Available: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']
task = protocol['task']
primary_metric = task.get('primary_metric') or protocol.get('primary_metric')
if primary_metric is None:
    raise ValueError(f"Missing primary_metric. Available task keys: {list(task.keys())}, protocol keys: {list(protocol.keys())}")

train_df = pd.read_parquet(inputs['train'])
test_df = pd.read_parquet(inputs['test'])

exclude_set = set(targets) | set(excluded) | ({group_col} if group_col else set())
candidate_cols = [c for c in train_df.columns if c not in exclude_set]
feature_cols = []
for c in candidate_cols:
    s = train_df[c]
    if (s.dtype == 'object' or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
        continue
    feature_cols.append(c)
feature_cols = [c for c in feature_cols if c not in exclude_set]

val_metrics = pd.read_csv(inputs['09_validate_score_validation_8SYXFK0A/validation_metrics.csv'])
if 'metric' in val_metrics.columns and 'value' in val_metrics.columns:
    sub = val_metrics[val_metrics['metric'] == primary_metric]
    if sub.empty:
        raise ValueError(f"No validation rows for metric {primary_metric}")
    best_row = sub.loc[sub['value'].idxmin()]
    best_model_name = best_row['model']
elif primary_metric in val_metrics.columns:
    best_row = val_metrics.loc[val_metrics[primary_metric].idxmin()]
    best_model_name = best_row['model'] if 'model' in val_metrics.columns else best_row.name
else:
    raise ValueError(f"Validation metrics missing metric/value or {primary_metric} column")

name_lower = str(best_model_name).lower()
if 'baseline' in name_lower:
    model_path = inputs['07_model_train_baseline_0WM3R8E0/model.pkl']
elif 'alternative' in name_lower:
    model_path = inputs['08_model_train_alternative_B8FQJ4R0/alternative_model.pkl']
elif 'latest' in name_lower:
    model_path = inputs['latest_model']
else:
    model_path = None
    for k, v in inputs.items():
        if 'model' in k and name_lower in k.lower():
            model_path = v
            break
    if model_path is None:
        raise ValueError(f"Could not map model name {best_model_name} to a saved model path.")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

def get_feature_names(m):
    if hasattr(m, 'feature_names_in_'):
        return list(m.feature_names_in_)
    if hasattr(m, 'named_steps'):
        for step in m.named_steps.values():
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return None

expected = get_feature_names(model)
if expected is not None:
    use_cols = [c for c in expected if c in test_df.columns]
    if len(use_cols) != len(expected):
        missing = set(expected) - set(test_df.columns)
        raise ValueError(f"Missing expected columns: {missing}")
    bad = set(use_cols) & (set(targets) | set(excluded) | ({group_col} if group_col else set()))
    if bad:
        raise ValueError(f"Model expects target/excluded columns: {bad}")
else:
    use_cols = feature_cols

missing = set(use_cols) - set(test_df.columns)
if missing:
    raise ValueError(f"Missing feature columns in test: {missing}")

X_test = test_df[use_cols].copy()
y_pred = model.predict(X_test)
y_true = test_df[targets].values
if y_pred.ndim == 1:
    y_pred = y_pred.reshape(-1, 1)

rows = []
for i, target in enumerate(targets):
    yt = y_true[:, i]
    yp = y_pred[:, i]
    rows.append({'model': best_model_name, 'metric': f'rmse_{target}', 'value': np.sqrt(mean_squared_error(yt, yp))})
    rows.append({'model': best_model_name, 'metric': f'mae_{target}', 'value': mean_absolute_error(yt, yp)})
    rows.append({'model': best_model_name, 'metric': f'r2_{target}', 'value': r2_score(yt, yp)})

rmse_macro = np.mean([r['value'] for r in rows if r['metric'].startswith('rmse_')])
mae_macro = np.mean([r['value'] for r in rows if r['metric'].startswith('mae_')])
r2_macro = np.mean([r['value'] for r in rows if r['metric'].startswith('r2_')])
rows.append({'model': best_model_name, 'metric': 'rmse_macro', 'value': rmse_macro})
rows.append({'model': best_model_name, 'metric': 'mae_macro', 'value': mae_macro})
rows.append({'model': best_model_name, 'metric': 'r2_macro', 'value': r2_macro})

df_out = pd.DataFrame(rows)
df_out['value'] = df_out['value'].apply(lambda x: float(f"{x:.4g}"))
mlca_runtime.save_table(df_out, "test_metrics.csv", caption="Test metrics per target and macro average")

mlca_runtime.record_metric("rmse", rmse_macro, "test")
mlca_runtime.record_fact_candidate("test_score", rmse_macro, source="test_metrics.csv")

print(f"Test rows: {len(test_df)}")
print(f"Selected model: {best_model_name}")
print(f"Test RMSE macro: {rmse_macro:.4f}")
print("Saved test_metrics.csv")
