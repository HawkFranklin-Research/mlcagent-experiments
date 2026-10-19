import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
for k in ['targets', 'excluded_columns', 'task']:
    if k not in protocol:
        raise KeyError(f"Missing {k}. Available top-level keys: {list(protocol.keys())}")

task = protocol['task']
primary = task['primary_metric']
targets = list(protocol['targets'])
excluded = set(protocol['excluded_columns'])
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

def is_identifier(s):
    if s.nunique(dropna=False) != len(s):
        return False
    return not pd.api.types.is_float_dtype(s)

feat_cols = [
    c for c in train.columns
    if c not in set(targets) and c not in excluded and c != group
    and not is_identifier(train[c])
]

X_val = val[feat_cols]
y_val = val[targets[0]]

def load_name(path, default):
    try:
        m = json.loads(inputs[path].read_text())
        return m.get('name') or m.get('model_name') or default
    except Exception:
        return default

name_log = load_name('07_model_train_logistic_61NSXR0X/manifest.json', 'logistic')
name_alt = load_name('08_model_train_alternative_K9A47CPR/manifest.json', 'alternative')

with open(inputs['07_model_train_logistic_61NSXR0X/model.pkl'], 'rb') as f:
    m_log = pickle.load(f)
with open(inputs['08_model_train_alternative_K9A47CPR/alternative_model.pkl'], 'rb') as f:
    m_alt = pickle.load(f)

def macro_ovr_auc(y_true, proba, classes):
    present = np.unique(y_true)
    if len(present) == len(classes):
        return roc_auc_score(y_true, proba, multi_class='ovr', average='macro', labels=classes)
    aucs = []
    for c in present:
        j = list(classes).index(c)
        aucs.append(roc_auc_score((y_true == c).astype(int), proba[:, j]))
    missing = set(classes) - set(present)
    print(f"AUC classes missing: {missing}")
    return float(np.mean(aucs))

def evaluate(model, name):
    if hasattr(model, 'feature_names_in_'):
        cols = list(model.feature_names_in_)
        missing = set(cols) - set(X_val.columns)
        if missing:
            raise KeyError(f"Model {name} expects missing columns: {missing}")
        Xm = X_val[cols]
    else:
        Xm = X_val
    proba = model.predict_proba(Xm)
    auc = macro_ovr_auc(y_val, proba, model.classes_)
    pred = model.predict(Xm)
    f1 = f1_score(y_val, pred, average='macro')
    return auc, f1

rows = []
for model, name in [(m_log, name_log), (m_alt, name_alt)]:
    auc, f1 = evaluate(model, name)
    rows.append({'model': name, 'metric': primary, 'value': auc, 'threshold': np.nan})
    rows.append({'model': name, 'metric': 'macro_f1', 'value': f1, 'threshold': np.nan})

metrics_df = pd.DataFrame(rows)
metrics_df['value'] = metrics_df['value'].round(4)
mlca_runtime.save_table(metrics_df, 'validation_metrics.csv')

primary_rows = metrics_df[metrics_df['metric'] == primary]
best_idx = primary_rows['value'].idxmax()
best_row = metrics_df.loc[best_idx]
best_name = best_row['model']
best_val = float(best_row['value'])

mlca_runtime.record_metric(primary, best_val, split='val')
mlca_runtime.record_fact_candidate('best_model', best_name, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_roc_auc', best_val, source='validation_metrics.csv')

print(f"train {train.shape}, val {val.shape}, features {len(feat_cols)}")
print(f"best model: {best_name} ({primary}={best_val:.4f})")
