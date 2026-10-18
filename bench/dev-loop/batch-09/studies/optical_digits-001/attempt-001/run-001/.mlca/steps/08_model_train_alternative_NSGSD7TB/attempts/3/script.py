import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

pre = load(inputs['05_preprocess_fit_preprocessing_ZVECC4HC/preprocessor.pkl'])
sel = load(inputs['06_features_retain_features_SAGMFV47/selector_all.pkl'])

def cols_of(o):
    c = getattr(o, 'feature_names_in_', None)
    return list(c) if c is not None else None

pre_cols, sel_cols = cols_of(pre), cols_of(sel)
drop = set(targets) | set(excluded) | ({group} if group else set())

wraps = bool(isinstance(sel, Pipeline) and pre_cols and sel_cols and set(pre_cols) == set(sel_cols))
if wraps:
    frontend, feature_cols = sel, list(sel_cols)
else:
    feature_cols = list(pre_cols) if pre_cols else [c for c in train.columns if c not in drop]
    frontend = Pipeline([('preprocessor', pre), ('selector', sel)])
feature_cols = [c for c in feature_cols if c in train.columns]
assert not (set(feature_cols) & drop), f"leak: {set(feature_cols) & drop}"

X_train = train[feature_cols]
X_val = val[feature_cols]

def enable_pandas(e):
    if hasattr(e, 'set_output'):
        try: e.set_output(transform='pandas')
        except Exception: pass
    if isinstance(e, Pipeline):
        for _, s in e.steps: enable_pandas(s)
enable_pandas(frontend)

Z_train = frontend.transform(X_train)
if not isinstance(Z_train, pd.DataFrame):
    Z_train = pd.DataFrame(np.asarray(Z_train, dtype=float), index=X_train.index,
                           columns=[f'z{i}' for i in range(np.asarray(Z_train).shape[1])])

mlp = MLPRegressor(hidden_layer_sizes=(16, 8, 16), max_iter=300, random_state=42)
pipe = Pipeline([('frontend', frontend), ('mlp', mlp)])
mlca_runtime.fit_record(pipe, X_train, 'train', y=Z_train)

Z_val = frontend.transform(X_val)
if not isinstance(Z_val, pd.DataFrame):
    Z_val = pd.DataFrame(np.asarray(Z_val, dtype=float), index=X_val.index, columns=Z_train.columns)

def rec_mse(p, X, Zref):
    P = np.asarray(p.predict(X), dtype=float)
    return float(mean_squared_error(np.asarray(Zref, dtype=float), P))

tr = rec_mse(pipe, X_train, Z_train)
va = rec_mse(pipe, X_val, Z_val)
mlca_runtime.record_metric('reconstruction_error', tr, 'train')
mlca_runtime.record_metric('reconstruction_error', va, 'val')
mlca_runtime.record_metric('embedding_dim', 8.0, 'train')
mlca_runtime.record_metric('n_processed_features', float(Z_train.shape[1]), 'train')
mlca_runtime.save_model(pipe, 'alternative_model.pkl')
print(f"rows={len(X_train)} feats={len(feature_cols)} processed={Z_train.shape[1]}")
print(f"train_mse={tr:.4f} val_mse={va:.4f}")
