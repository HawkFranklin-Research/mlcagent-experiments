import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
    balanced_accuracy_score, brier_score_loss, f1_score, roc_curve)
from sklearn.pipeline import Pipeline
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
for k in ('task', 'targets', 'excluded_columns'):
    if k not in proto:
        raise KeyError(f"Missing '{k}'. Available top-level keys: {list(proto.keys())}")

task = proto['task']
targets = proto['targets']
excluded = proto.get('excluded_columns', [])
group_col = proto.get('group_column', None)
primary = task['primary_metric']
pos_label = proto.get('positive_label', None)

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

def keylike(s):
    if s.nunique() != len(s):
        return False
    if pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_integer_dtype(s):
            try:
                return (s.min() >= 0) and (s.max() - s.min()) == len(s) - 1
            except Exception:
                return False
        return False
    return True

feature_cols = [c for c in train.columns
                if c not in targets and c not in excluded and c != group_col]
feature_cols = [c for c in feature_cols if not keylike(train[c])]

with open(inp['05_preprocess_fit_preprocessing_2E873D38/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inp['06_features_select_features_CMGHE25G/selector.pkl'], 'rb') as f:
    selector = pickle.load(f)
models = {}
with open(inp['07_model_train_logistic_Q726PM5C/model.pkl'], 'rb') as f:
    models['07_logistic'] = pickle.load(f)
with open(inp['08_model_train_alternative_19QA2ANC/alternative_model.pkl'], 'rb') as f:
    models['08_alternative'] = pickle.load(f)

def get_proba(model, df):
    cols = list(getattr(model, 'feature_names_in_', []))
    if cols and all(c in df.columns for c in cols):
        if isinstance(model, Pipeline) or len(cols) == len(feature_cols):
            return model.predict_proba(df[cols])
    X = df[feature_cols]
    if hasattr(preprocessor, 'transform'):
        X = preprocessor.transform(X)
    if selector is not None and hasattr(selector, 'transform'):
        X = selector.transform(X)
    return model.predict_proba(X)

y_col = targets[0]
y_val = val[y_col]

rows = []
for name, model in models.items():
    proba = get_proba(model, val)
    if hasattr(proba, 'values'):
        proba = proba.values
    proba = np.asarray(proba)
    classes = np.array(list(model.classes_))
    if len(classes) == 2:
        j = classes.tolist().index(pos_label) if (pos_label is not None and pos_label in classes.tolist()) else 1
        p_pos = proba[:, j]
        y_bin = (y_val == classes[j]).astype(int).values
        auc = float(roc_auc_score(y_bin, p_pos))
        pr = float(average_precision_score(y_bin, p_pos))
        brier = float(brier_score_loss(y_bin, p_pos))
        fpr, tpr, thr = roc_curve(y_bin, p_pos)
        ythr = float(thr[np.argmax(tpr - fpr)])
        bal = float(balanced_accuracy_score(y_bin, (p_pos >= ythr).astype(int)))
        rows += [
            {'model': name, 'metric': 'roc_auc', 'value': auc, 'threshold': np.nan},
            {'model': name, 'metric': 'pr_auc', 'value': pr, 'threshold': np.nan},
            {'model': name, 'metric': 'balanced_accuracy', 'value': bal, 'threshold': ythr},
            {'model': name, 'metric': 'brier_score', 'value': brier, 'threshold': np.nan},
        ]
    else:
        y_true = y_val.values
        present = [c for c in classes if (y_true == c).any()]
        if len(present) == len(classes):
            ovr = float(roc_auc_score(y_true, proba, multi_class='ovr',
                                      average='macro', labels=classes.tolist()))
        else:
            ovr = float(np.mean([roc_auc_score((y_true == c).astype(int),
                                proba[:, list(classes).index(c)]) for c in present]))
        pred = classes[np.argmax(proba, axis=1)]
        f1m = float(f1_score(y_true, pred, average='macro'))
        rows += [
            {'model': name, 'metric': 'roc_auc_ovr', 'value': ovr, 'threshold': np.nan},
            {'model': name, 'metric': 'f1_macro', 'value': f1m, 'threshold': np.nan},
        ]

metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, 'validation_metrics.csv', 'Validation metrics per model')

primary_vals = metrics_df[metrics_df['metric'] == primary].copy()
lower_better = primary in ('brier_score', 'log_loss', 'rmse', 'mae')
best_idx = primary_vals['value'].idxmin() if lower_better else primary_vals['value'].idxmax()
best_row = primary_vals.loc[best_idx]

for _, r in primary_vals.iterrows():
    mlca_runtime.record_metric(f"{primary}_{r['model']}", float(r['value']), split='val')

mlca_runtime.record_fact_candidate('best_model', best_row['model'],
                                   source='validation_metrics.csv')
if primary == 'roc_auc':
    mlca_runtime.record_fact_candidate('best_val_roc_auc', float(best_row['value']),
                                       source='validation_metrics.csv')

print(f"val_rows={len(val)} features={len(feature_cols)} models={list(models)}")
print(f"primary={primary} best={best_row['model']} val={float(best_row['value']):.4f}")
