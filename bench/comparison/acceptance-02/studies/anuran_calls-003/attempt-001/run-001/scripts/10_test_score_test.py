import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
import cloudpickle
from sklearn.metrics import f1_score, hamming_loss
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs['_protocol'].read_text())
targets = list(proto['targets'])
excluded = list(proto.get('excluded_columns', []) or [])
group = proto.get('group_column')
primary = proto['task']['primary_metric']

test = pd.read_parquet(inputs['test'])

val_df = pd.read_csv(inputs['09_validate_score_validation_DX8R1YJ4/validation_metrics.csv'])
cols = list(val_df.columns)
mcol = next((c for c in cols if 'metric' in c.lower()), cols[1])
vcol = next((c for c in cols if 'value' in c.lower()), cols[-1])
modcol = next((c for c in cols if 'model' in c.lower()), cols[0])
pv = val_df[val_df[mcol].astype(str) == str(primary)]
if pv.empty:
    raise ValueError(f"primary '{primary}' absent: cols={cols}")
best_row = pv.loc[pv[vcol].astype(float).idxmax()]
best_model_name = str(best_row[modcol])

bm = best_model_name.lower()
if 'baseline' in bm or '07' in bm:
    model_path = inputs['07_model_train_baseline_8KXAGBNQ/model.pkl']
else:
    model_path = inputs['08_model_train_alternative_K6DZGSFY/alternative_model.pkl']

def load(path):
    with open(path, 'rb') as f:
        try:
            return cloudpickle.load(f)
        except Exception:
            f.seek(0)
            return pickle.load(f)

model = load(model_path)

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
raw_feats = [c for c in test.columns if c not in drop]

fn = getattr(model, 'feature_names_in_', None)
mfeat = list(fn) if fn is not None else []

if mfeat and set(mfeat).issubset(set(test.columns)) and not set(mfeat) & drop:
    X_test = test[mfeat]
    predict_model = model
else:
    pre = None
    sel = None
    try:
        pre = load(inputs['05_preprocess_fit_preprocessing_7F4HZTDJ/preprocessor.pkl'])
    except Exception:
        pre = None
    try:
        sel = load(inputs['06_features_retain_features_2RG0T0WD/selector_all.pkl'])
    except Exception:
        sel = None
    frontend = None
    for cand in (sel, pre):
        if cand is None:
            continue
        cf = getattr(cand, 'feature_names_in_', None)
        cfl = list(cf) if cf is not None else []
        if not cfl or set(cfl).issubset(set(raw_feats)):
            frontend = cand
            break
    if frontend is None and pre is not None and sel is not None:
        frontend = Pipeline([('pre', pre), ('sel', sel)])
    if frontend is None:
        raise ValueError(f"no usable frontend for model inputs {mfeat[:5]}...")
    predict_model = Pipeline([('frontend', frontend), ('model', model)])
    ff = getattr(frontend, 'feature_names_in_', None)
    feat_used = list(ff) if ff is not None else raw_feats
    X_test = test[feat_used]

pred = predict_model.predict(X_test)
y_true = test[targets].reset_index(drop=True)
y_pred = pd.DataFrame(np.asarray(pred), columns=targets)

f1s, hams = [], []
for t in targets:
    yt = np.asarray(y_true[t])
    yp = np.asarray(y_pred[t])
    f1s.append(f1_score(yt, yp, average='macro', zero_division=0))
    hams.append(hamming_loss(yt, yp))
macro_f1 = float(np.mean(f1s))
ham = float(np.mean(hams))
subset_acc = float((y_true.values == y_pred.values).all(axis=1).mean())

rows = [
    {'model': best_model_name, 'metric': primary, 'value': round(macro_f1, 6)},
    {'model': best_model_name, 'metric': 'hamming_loss', 'value': round(ham, 6)},
    {'model': best_model_name, 'metric': 'subset_accuracy', 'value': round(subset_acc, 6)},
]
mlca_runtime.save_table(pd.DataFrame(rows), 'test_metrics.csv', 'Test metrics (frozen candidate)')
mlca_runtime.record_metric(primary, macro_f1, 'test')
mlca_runtime.record_metric('hamming_loss', ham, 'test')
mlca_runtime.record_metric('subset_accuracy', subset_acc, 'test')
mlca_runtime.record_fact_candidate('test_score', macro_f1, source='test_metrics.csv')
print(f"selected={best_model_name} {primary}={macro_f1:.4f} ham={ham:.4f} sub={subset_acc:.4f} n={len(test)}")
