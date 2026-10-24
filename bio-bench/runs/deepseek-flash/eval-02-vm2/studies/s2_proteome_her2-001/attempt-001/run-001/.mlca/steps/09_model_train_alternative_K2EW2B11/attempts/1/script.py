import mlca_runtime, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

SEED = 42
mlca_runtime.set_seed(SEED)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
task = protocol['task']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

cands = [inputs['08_model_train_logistic_9HTWE67Q/model.pkl'],
         inputs['latest_model'],
         inputs['07_features_select_features_C8HA72DE/selector.pkl'],
         inputs['06_preprocess_fit_preprocessing_R9QMSZCE/preprocessor.pkl']]
best = None
for p in cands:
    p = Path(p)
    if not p.exists():
        continue
    try:
        obj = load_pkl(p)
    except Exception:
        continue
    if isinstance(obj, Pipeline) and hasattr(obj, 'feature_names_in_'):
        if best is None or len(obj.steps) > len(best.steps):
            best = obj

if best is None:
    raise RuntimeError('No usable saved pipeline found')

feat_cols = list(best.feature_names_in_)
steps = list(best.steps)
if hasattr(steps[-1][1], 'predict_proba'):
    steps = steps[:-1]
steps = steps + [('hist_gb', HistGradientBoostingClassifier(random_state=SEED))]

pipe = Pipeline(steps=steps)

X_tr = train[feat_cols].copy()
y_tr = train[targets[0]].copy()

if y_tr.value_counts(normalize=True).min() < 0.10:
    pipe.set_params(hist_gb__class_weight='balanced')

pipe = mlca_runtime.fit_record(pipe, X_tr, 'train', y=y_tr)

# sanity check on the validation partition
y_va = val[targets[0]]
X_va = val[feat_cols].copy()
try:
    proba = pipe.predict_proba(X_va)
    est = pipe.named_steps['hist_gb']
    pos_lbl = est.classes_[1]
    y_bin = (y_va == pos_lbl).astype(int)
    auc = roc_auc_score(y_bin, proba[:, 1])
    mlca_runtime.record_metric('roc_auc', float(auc), 'val')
    print('val_roc_auc', round(auc, 4))
except Exception as e:
    print('val_eval_error', str(e)[:120])

mlca_runtime.save_model(pipe, 'alternative_model.pkl')
print('n_train', len(X_tr), 'n_features', len(feat_cols))
