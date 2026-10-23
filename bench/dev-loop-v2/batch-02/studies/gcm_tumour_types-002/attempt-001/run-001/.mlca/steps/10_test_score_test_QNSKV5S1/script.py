import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for k in ['targets', 'task']:
    if k not in protocol:
        raise KeyError(f"Missing {k} in protocol. Available: {list(protocol.keys())}")
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column")
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing primary_metric in task. Available: {list(task.keys())}")
primary_metric = task["primary_metric"]

test = pd.read_parquet(inputs["test"])
all_cols = test.columns.tolist()
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]
key_cols = []
for c in feature_cols:
    if test[c].dtype.kind in ('O', 'i', 'u') and test[c].nunique() == len(test):
        key_cols.append(c)
feature_cols = [c for c in feature_cols if c not in key_cols]

# Determine selected model from validation
model_path = None
try:
    val_manifest = json.loads(inputs["09_validate_score_validation_G3N0P22Q/manifest.json"].read_text())
    for k in ['selected_model', 'best_model', 'selected', 'model_path', 'chosen_model']:
        if k in val_manifest:
            model_path = val_manifest[k]
            break
except Exception:
    pass
if model_path is None:
    try:
        val_metrics = pd.read_csv(inputs["09_validate_score_validation_G3N0P22Q/validation_metrics.csv"])
        if 'metric' in val_metrics.columns:
            pm = val_metrics[val_metrics['metric'] == primary_metric]
            if len(pm) > 0:
                higher_better = primary_metric in ['roc_auc', 'roc_auc_ovr', 'f1', 'macro_f1', 'accuracy', 'balanced_accuracy']
                idx = pm['value'].idxmax() if higher_better else pm['value'].idxmin()
                row = pm.loc[idx]
                model_name = row.get('model', row.get('candidate', None))
                if model_name:
                    if 'logistic' in model_name.lower():
                        model_path = inputs["07_model_train_logistic_N3PQQ99D/model.pkl"]
                    elif 'alternative' in model_name.lower():
                        model_path = inputs["08_model_train_alternative_SGYXE5GB/alternative_model.pkl"]
    except Exception:
        pass
if model_path is None:
    model_path = inputs["latest_model"]
if isinstance(model_path, str):
    if model_path in inputs:
        model_path = inputs[model_path]
    else:
        for k, v in inputs.items():
            if k.endswith(model_path):
                model_path = v
                break

with open(model_path, 'rb') as f:
    model = pickle.load(f)

if hasattr(model, 'feature_names_in_'):
    expected_cols = list(model.feature_names_in_)
elif hasattr(model, 'steps') and len(model.steps) > 0 and hasattr(model.steps[0][1], 'feature_names_in_'):
    expected_cols = list(model.steps[0][1].feature_names_in_)
else:
    expected_cols = feature_cols

X_test = test[expected_cols]
proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)
y_true = test[targets[0]].values
classes = list(model.classes_)

def macro_ovr_auc(y, p, cls):
    present = []
    for i, c in enumerate(cls):
        yb = (y == c).astype(int)
        if 0 < yb.sum() < len(yb):
            present.append(i)
    if not present:
        return np.nan
    return np.mean([roc_auc_score((y == cls[i]).astype(int), p[:, i]) for i in present])

primary_val = macro_ovr_auc(y_true, proba, classes)
macro_f1 = f1_score(y_true, y_pred, average='macro')

rng = np.random.default_rng(42)
n = len(y_true)
boot = []
for _ in range(1000):
    idx = rng.integers(0, n, n)
    v = macro_ovr_auc(y_true[idx], proba[idx, :], classes)
    if not np.isnan(v):
        boot.append(v)
if boot:
    ci_low, ci_high = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
else:
    ci_low = ci_high = np.nan

metrics_df = pd.DataFrame([
    {'metric': primary_metric, 'value': primary_val, 'ci_low': ci_low, 'ci_high': ci_high},
    {'metric': 'macro_f1', 'value': macro_f1, 'ci_low': np.nan, 'ci_high': np.nan}
])
metrics_df = metrics_df.round(4)
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics")
mlca_runtime.record_metric(primary_metric, primary_val, split='test', ci=(ci_low, ci_high))
mlca_runtime.record_fact_candidate('test_roc_auc', primary_val, source='test_metrics.csv')

pred_df = pd.DataFrame({
    'row_id': test.index,
    'partition': 'test',
    f'y_true__{targets[0]}': y_true,
    f'y_pred__{targets[0]}': y_pred,
})
for i, c in enumerate(classes):
    pred_df[f'proba__{c}'] = proba[:, i]
mlca_runtime.save_row_outputs(pred_df, "test_predictions.parquet")

missing = [c for c in classes if (y_true == c).sum() == 0]
if missing:
    print(f"Classes missing in test: {missing}")
print(f"Test rows: {len(test)}, features used: {len(expected_cols)}")
print(f"{primary_metric}: {primary_val:.4f} (CI {ci_low:.4f}-{ci_high:.4f})")
print(f"macro_f1: {macro_f1:.4f}")
