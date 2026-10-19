import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs['_protocol'].read_text())
targets = proto['targets']
excluded = proto['excluded_columns']
primary = proto['task']['primary_metric']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
test = pd.read_parquet(inputs['test'])

val_df = pd.read_csv(inputs['09_validate_score_validation_DX8R1YJ4/validation_metrics.csv'])
cols = list(val_df.columns)
mcol = 'metric' if 'metric' in cols else cols[1]
vcol = 'value' if 'value' in cols else cols[-1]
modcol = 'model' if 'model' in cols else cols[0]
pv = val_df[val_df[mcol] == primary]
if pv.empty:
    raise ValueError(f"primary metric '{primary}' absent from validation_metrics: {cols}")
best_row = pv.loc[pv[vcol].idxmax()]
best_model_name = str(best_row[modcol])
bm = best_model_name.lower()
if 'baseline' in bm or bm.startswith('07'):
    model_path = inputs['07_model_train_baseline_8KXAGBNQ/model.pkl']
else:
    model_path = inputs['08_model_train_alternative_K6DZGSFY/alternative_model.pkl']

with open(model_path, 'rb') as f:
    model = pickle.load(f)
feat = list(getattr(model, 'feature_names_in_', []) or [])
if not feat:
    feat = [c for c in train.columns
            if c not in targets and c not in excluded
            and c != proto.get('group_column')]
    nun = train[feat].nunique()
    feat = [c for c in feat if not (nun[c] == len(train) and
            (pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_integer_dtype(train[c])))]
X_test = test[feat]

pred = model.predict(X_test)
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
print(f"selected={best_model_name} {primary}={macro_f1:.4f} ham={ham:.4f} sub_acc={subset_acc:.4f} n_test={len(test)}")
