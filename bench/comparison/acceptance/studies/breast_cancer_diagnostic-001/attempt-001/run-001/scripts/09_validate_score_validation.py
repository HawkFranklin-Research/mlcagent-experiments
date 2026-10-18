import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, roc_curve, f1_score
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
task = proto['task']
targets = proto['targets']
excluded = set(proto.get('excluded_columns', []))
group = proto.get('group_column')

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
feature_cols = [c for c in train.columns if c not in set(targets) | excluded and c != group]
feature_cols = [
    c for c in feature_cols
    if not (train[c].dtype == object and train[c].nunique(dropna=False) == len(train))
]
X_val = val[feature_cols].copy()
y_val = val[targets[0]].copy()

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

pre = load_pkl(inp['05_preprocess_fit_preprocessing_WZN8YG48/preprocessor.pkl'])
sel = load_pkl(inp['06_features_select_features_6Y9H969T/selector.pkl'])

def transform(t, X):
    if t is None:
        return X
    fin = getattr(t, 'feature_names_in_', None)
    if fin is not None and hasattr(X, 'columns'):
        cols = [c for c in fin if c in X.columns]
        if len(cols) == len(fin):
            X = X[cols]
    return t.transform(X)

def model_input(model, X):
    if isinstance(model, Pipeline):
        names = [n.lower() for n in model.named_steps]
        if any('preprocess' in n or 'select' in n for n in names):
            return X
    fin = getattr(model, 'feature_names_in_', None)
    if fin is not None and list(fin) == list(X.columns):
        return X
    if isinstance(sel, Pipeline):
        snames = [n.lower() for n in sel.named_steps]
        if any('preprocess' in n for n in snames):
            return transform(sel, X)
    Xt = X
    used_sel = False
    try:
        Xt = transform(pre, Xt)
        used_sel = isinstance(pre, Pipeline) and any('select' in n.lower() for n in pre.named_steps)
    except Exception:
        pass
    if not used_sel:
        try:
            Xt = transform(sel, Xt)
        except Exception:
            pass
    return Xt

def score_model(model):
    Xin = model_input(model, X_val)
    proba = model.predict_proba(Xin)
    classes = list(getattr(model, 'classes_', []))
    if not classes:
        last = list(model.named_steps)[-1]
        classes = list(model.named_steps[last].classes_)
    classes = np.asarray(classes)
    if len(classes) <= 2:
        pos_label = task.get('positive_label', proto.get('positive_label'))
        if pos_label is None:
            pos_label = classes[1]
        if pos_label not in classes:
            pos_label = classes[1]
        pi = int(np.where(classes == pos_label)[0][0])
        p = proba[:, pi]
        yb = (y_val == pos_label).astype(int).to_numpy()
        fpr, tpr, thr = roc_curve(yb, p)
        j = tpr - fpr
        ok = np.isfinite(thr) & (thr >= 0) & (thr <= 1)
        th = float(thr[ok][np.argmax(j[ok])]) if ok.any() else 0.5
        yp = (p >= th).astype(int)
        metrics = {
            'roc_auc': roc_auc_score(yb, p),
            'pr_auc': average_precision_score(yb, p),
            'balanced_accuracy': balanced_accuracy_score(yb, yp),
            'brier_score': brier_score_loss(yb, p),
        }
        thresholds = {'balanced_accuracy': th}
    else:
        yp = model.predict(Xin)
        metrics = {
            'roc_auc': roc_auc_score(y_val, proba, multi_class='ovr', average='macro', labels=classes),
            'f1_macro': f1_score(y_val, yp, average='macro'),
        }
        thresholds = {}
    return metrics, thresholds

cands = {
    'logistic': load_pkl(inp['07_model_train_logistic_BMNCC8GS/model.pkl']),
    'alternative': load_pkl(inp['08_model_train_alternative_KFV9SB0C/alternative_model.pkl']),
}
primary = task['primary_metric']
rows = []
by_model = {}
for name, model in cands.items():
    metrics, thresholds = score_model(model)
    by_model[name] = metrics
    if primary in metrics:
        mlca_runtime.record_metric(primary, float(metrics[primary]), 'val')
    for metric, value in metrics.items():
        rows.append({
            'model': name,
            'metric': metric,
            'value': float(value),
            'threshold': thresholds.get(metric, np.nan),
        })

higher = primary not in {'brier_score', 'log_loss'}
vals = {n: by_model[n][primary] for n in by_model if primary in by_model[n]}
if vals:
    best = max(vals, key=vals.get) if higher else min(vals, key=vals.get)
else:
    best = next(iter(cands))
    vals = {best: np.nan}

def sig(x):
    return float(f"{x:.4g}") if pd.notna(x) else np.nan

for row in rows:
    row['value'] = sig(row['value'])
    row['threshold'] = sig(row['threshold'])
df = pd.DataFrame(rows, columns=['model', 'metric', 'value', 'threshold'])
mlca_runtime.save_table(df, 'validation_metrics.csv', caption='Validation metrics')

mlca_runtime.record_fact_candidate('best_model', best, source='validation_metrics.csv')
if not np.isnan(vals.get(best, np.nan)) and 'roc_auc' in by_model[best]:
    mlca_runtime.record_fact_candidate(
        'best_val_roc_auc', float(by_model[best]['roc_auc']), source='validation_metrics.csv'
    )

print(f"train={len(train)} val={len(val)} features={len(feature_cols)}")
print(f"primary={primary} best={best} value={vals.get(best, np.nan):.4g}")
