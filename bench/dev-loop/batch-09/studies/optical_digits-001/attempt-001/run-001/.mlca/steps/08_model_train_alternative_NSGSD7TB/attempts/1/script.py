import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

with open(inputs['_protocol']) as f:
    protocol = json.load(f)
targets, excluded, group = protocol['targets'], protocol['excluded_columns'], protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

preprocessor = load(inputs['05_preprocess_fit_preprocessing_ZVECC4HC/preprocessor.pkl'])
selector_all = load(inputs['06_features_retain_features_SAGMFV47/selector_all.pkl'])

def get_cols(o):
    return list(o.feature_names_in_) if hasattr(o, 'feature_names_in_') else None

pre_cols, sel_cols = get_cols(preprocessor), get_cols(selector_all)

def wraps_pre(sel, pre):
    if not isinstance(sel, Pipeline):
        return False
    sc, pc = getattr(sel, 'feature_names_in_', None), getattr(pre, 'feature_names_in_', None)
    return sc is not None and pc is not None and set(sc) == set(pc)

drop = set(targets) | set(excluded) | ({group} if group else set())
if wraps_pre(selector_all, preprocessor):
    frontend = selector_all
else:
    frontend = Pipeline([('preprocessor', preprocessor), ('selector', selector_all)])

feature_cols = pre_cols or sel_cols or [c for c in train.columns if c not in drop]
feature_cols = [c for c in feature_cols if c in train.columns]
assert not (set(feature_cols) & drop), f"leak: {set(feature_cols) & drop}"

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()

# processed features as reconstruction targets, computed with prefit frontend
proc = frontend.transform(X_train)
proc = proc.toarray() if hasattr(proc, 'toarray') else np.asarray(proc, dtype=float)
y_train = np.asarray(proc, dtype=float)
ncols = y_train.shape[1]

if ncols <= 300:
    def enable_pandas(e):
        if hasattr(e, 'set_output'):
            try:
                e.set_output(transform='pandas')
            except Exception:
                pass
        if isinstance(e, Pipeline):
            for _, s in e.steps:
                enable_pandas(s)
    enable_pandas(frontend)

mlp = MLPRegressor(hidden_layer_sizes=(16, 8, 16), max_iter=300, random_state=42)
pipe = Pipeline([('frontend', frontend), ('mlp', mlp)])
mlca_runtime.fit_record(pipe, X_train, 'train', y=y_train)

def rec_err(p, Xr):
    t = p.named_steps['frontend'].transform(Xr)
    t = t.toarray() if hasattr(t, 'toarray') else np.asarray(t, dtype=float)
    return float(mean_squared_error(np.asarray(t, dtype=float), p.predict(Xr)))

tr_err, va_err = rec_err(pipe, X_train), rec_err(pipe, X_val)
mlca_runtime.record_metric('reconstruction_error', tr_err, 'train')
mlca_runtime.record_metric('reconstruction_error', va_err, 'val')
mlca_runtime.record_metric('embedding_dim', 8.0, 'train')
mlca_runtime.record_metric('n_processed_features', float(ncols), 'train')

mlca_runtime.save_model(pipe, 'alternative_model.pkl')

print(f"train rows={len(X_train)} feature_cols={len(feature_cols)} processed_width={ncols}")
print(f"train_reconstruction_mse={tr_err:.4f}")
print(f"val_reconstruction_mse={va_err:.4f}")
