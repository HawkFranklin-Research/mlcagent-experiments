import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

def key_like(s):
    if len(s) <= 1:
        return False
    if s.dtype == object:
        return s.nunique(dropna=True) == s.notna().sum()
    if s.dtype.kind in 'iu':
        return s.nunique(dropna=True) == len(s)
    return False

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feature_cols = [c for c in train.columns if c not in drop]
feature_cols = [c for c in feature_cols if not key_like(train[c])]

with open(inputs['05_preprocess_fit_preprocessing_E2NYE0HC/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

if hasattr(pre, 'named_steps'):
    steps = list(pre.named_steps)
else:
    steps = []
if hasattr(pre, 'feature_names_in_'):
    pre_cols = list(pre.feature_names_in_)
    missing = [c for c in pre_cols if c not in feature_cols]
    if missing:
        raise ValueError(f'preprocessor expects missing columns: {missing}')
    feature_cols = pre_cols

Xtr = train[feature_cols]
Xva = val[feature_cols]

pipe = Pipeline([
    ('pre', pre),
    ('mlp', MLPRegressor(hidden_layer_sizes=(16, 8, 16),
                         max_iter=300, random_state=42)),
])

Ytr = np.asarray(pre.transform(Xtr))
Yva = np.asarray(pre.transform(Xva))

mlca_runtime.fit_record(pipe, Xtr, partition='train', y=Ytr)

pred_tr = np.asarray(pipe.predict(Xtr))
pred_va = np.asarray(pipe.predict(Xva))
if hasattr(Ytr, 'toarray'):
    Ytr = Ytr.toarray(); Yva = Yva.toarray()

err_tr = float(np.mean((pred_tr - Ytr) ** 2))
err_va = float(np.mean((pred_va - Yva) ** 2))

mlca_runtime.record_metric('reconstruction_error', err_tr, 'train')
mlca_runtime.record_metric('reconstruction_error', err_va, 'val')
mlca_runtime.save_model(pipe, 'alternative_model.pkl')

print(f'features={len(feature_cols)} processed_dim={Ytr.shape[1]}')
print(f'train_recon_mse={err_tr:.4g}')
print(f'val_recon_mse={err_va:.4g}')
