import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
proto = json.loads(inputs['_protocol'].read_text())

for key in ['targets', 'excluded_columns']:
    if key not in proto:
        raise KeyError(f"Missing '{key}' in protocol. Available keys: {list(proto.keys())}")
targets = proto['targets']
excluded = proto['excluded_columns']
group = proto.get('group_column', None)
if 'task' not in proto:
    raise KeyError(f"Missing 'task' in protocol. Available keys: {list(proto.keys())}")
task = proto['task']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['05_preprocess_fit_preprocessing_JYNCWEGN/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_retain_features_HY561TZJ/selector_all.pkl'], 'rb') as f:
    sel = pickle.load(f)

def get_expected_cols(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            cols = get_expected_cols(step)
            if cols is not None:
                return cols
    return None

expected_sel = get_expected_cols(sel)
bad = set(targets + excluded + ([group] if group else []))
if expected_sel is not None and all(c in train.columns for c in expected_sel) and not (set(expected_sel) & bad):
    frontend = sel
else:
    frontend = Pipeline([('preprocessor', pre), ('selector', sel)])

expected = get_expected_cols(frontend)
if expected is not None:
    feature_cols = [c for c in expected if c in train.columns and c not in bad]
else:
    feature_cols = [c for c in train.columns if c not in bad]
    keep = []
    for c in feature_cols:
        s = train[c]
        if (s.dtype == 'object' or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
            continue
        if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
            continue
        keep.append(c)
    feature_cols = keep

n_comp = min(8, len(feature_cols))
pca = PCA(n_components=n_comp, random_state=42)
full = Pipeline([('frontend', frontend), ('pca', pca)])

X_train = train[feature_cols]
X_val = val[feature_cols]

mlca_runtime.fit_record(full, X_train, partition='train')

frontend_fitted = full.named_steps['frontend']
pca_fitted = full.named_steps['pca']

def to_np(X):
    if hasattr(X, 'toarray'):
        return X.toarray()
    if hasattr(X, 'values'):
        return X.values
    return np.asarray(X)

def recon_err(X):
    Xf = frontend_fitted.transform(X)
    Xp = pca_fitted.transform(Xf)
    Xr = pca_fitted.inverse_transform(Xp)
    return float(((to_np(Xf) - to_np(Xr))**2).mean())

train_err = recon_err(X_train)
val_err = recon_err(X_val)

mlca_runtime.record_metric('reconstruction_error', train_err, 'train')
mlca_runtime.record_metric('reconstruction_error', val_err, 'val')

mlca_runtime.save_model(full, 'model.pkl')

print(f"features: {len(feature_cols)}, pca_components: {n_comp}")
print(f"train_recon_error: {train_err:.4f}")
print(f"val_recon_error: {val_err:.4f}")
