import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import scipy.special as sp

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
primary_metric = protocol['task']['primary_metric']
targets = protocol.get('targets', [])
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
test = pd.read_parquet(inputs['test'])

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

preproc = load_pkl(inputs['05_preprocess_fit_preprocessing_XD81KYMB/preprocessor.pkl'])
selector = load_pkl(inputs['06_features_retain_features_75C8096C/selector_all.pkl'])
model_base = load_pkl(inputs['07_model_train_baseline_93EYVTD0/model.pkl'])
model_alt = load_pkl(inputs['08_model_train_alternative_NZ42FXM6/alternative_model.pkl'])

def split_model(model):
    if isinstance(model, Pipeline):
        steps = model.steps
        if len(steps) == 1:
            return None, steps[0][1]
        return Pipeline(steps[:-1]), steps[-1][1]
    return None, model

def get_input_cols(obj):
    if obj is None:
        return None
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        first_name = obj.steps[0][0]
        first_step = obj.named_steps[first_name]
        if hasattr(first_step, 'feature_names_in_'):
            return list(first_step.feature_names_in_)
    return None

feature_cols = None
for obj in [preproc, selector, model_base, model_alt]:
    cols = get_input_cols(obj)
    if cols is not None:
        feature_cols = cols
        break
if feature_cols is None:
    cols = [c for c in train.columns if c not in targets and c not in excluded]
    if group_col in cols:
        cols.remove(group_col)
    drop = []
    for c in cols:
        s = train[c]
        if s.dtype == object or (pd.api.types.is_integer_dtype(s) and s.nunique() == len(s)):
            drop.append(c)
    feature_cols = [c for c in cols if c not in drop]

def build_frontend(preproc, selector):
    if isinstance(selector, Pipeline):
        for step in selector.named_steps.values():
            if step is preproc:
                return selector
        first_step = selector.named_steps[selector.steps[0][0]]
        if hasattr(first_step, 'transformers'):
            return selector
        return Pipeline([('preproc', preproc), ('selector', selector)])
    return Pipeline([('preproc', preproc), ('selector', selector)])

canonical_front = build_frontend(preproc, selector)

def get_frontend_and_est(model, canonical_front):
    front, est = split_model(model)
    if front is None:
        front = canonical_front
    return front, est

def compute_bottleneck(mlp, X):
    hidden = mlp.hidden_layer_sizes
    if isinstance(hidden, int):
        hidden = (hidden,)
    min_size = min(hidden)
    idxs = [i for i, s in enumerate(hidden) if s == min_size]
    mid = len(hidden) // 2
    bidx = min(idxs, key=lambda i: abs(i - mid))
    A = np.asarray(X, dtype=float)
    act = getattr(mlp, 'activation', 'relu')
    for i in range(bidx + 1):
        z = A @ mlp.coefs_[i] + mlp.intercepts_[i]
        if act == 'relu':
            A = np.maximum(z, 0)
        elif act == 'tanh':
            A = np.tanh(z)
        elif act == 'logistic':
            A = sp.expit(z)
        elif act == 'identity':
            A = z
        else:
            raise ValueError(f'Unknown activation {act}')
    return A

def compute_mse_and_embedding(front, est, X_raw, kind, need_embedding=False):
    input_cols = get_input_cols(front)
    if input_cols is None:
        input_cols = [c for c in X_raw.columns if c in feature_cols]
    X_in = X_raw[input_cols] if input_cols else X_raw
    if front is not None:
        X_front = front.transform(X_in)
    else:
        X_front = X_in
    if hasattr(X_front, 'toarray'):
        X_front = X_front.toarray()
    X_arr = np.asarray(X_front, dtype=float)
    if kind == 'pca':
        X_rec = est.inverse_transform(est.transform(X_arr))
        emb = est.transform(X_arr) if need_embedding else None
    elif kind == 'mlp':
        X_rec = est.predict(X_arr)
        emb = compute_bottleneck(est, X_arr) if need_embedding else None
    else:
        raise ValueError(f'Unknown kind {kind}')
    mse = float(np.mean((X_arr - X_rec) ** 2))
    return mse, emb

candidates = []
for model in [model_base, model_alt]:
    front, est = get_frontend_and_est(model, canonical_front)
    if isinstance(est, PCA):
        kind, name = 'pca', 'PCA'
    elif isinstance(est, MLPRegressor):
        kind, name = 'mlp', 'MLPRegressor'
    else:
        raise ValueError(f'Unknown estimator {type(est)}')
    mse, _ = compute_mse_and_embedding(front, est, val, kind, need_embedding=False)
    candidates.append({'name': name, 'front': front, 'est': est, 'kind': kind, 'val_mse': mse})

best = min(candidates, key=lambda d: d['val_mse'])
sel_name = best['name']
sel_front = best['front']
sel_est = best['est']
sel_kind = best['kind']

test_mse, _ = compute_mse_and_embedding(sel_front, sel_est, test, sel_kind, need_embedding=False)

metrics_df = pd.DataFrame([{'model': sel_name, 'metric': primary_metric, 'value': round(test_mse, 4)}])
mlca_runtime.save_table(metrics_df, 'test_metrics.csv', caption='Test reconstruction metric for selected model')
mlca_runtime.record_metric(primary_metric, test_mse, 'test')
mlca_runtime.record_fact_candidate('test_score', test_mse, source='test_metrics.csv')

rows = []
for part, df in [('train', train), ('test', test)]:
    _, emb = compute_mse_and_embedding(sel_front, sel_est, df, sel_kind, need_embedding=True)
    if emb is None:
        raise RuntimeError('Embedding computation failed')
    n_comp = emb.shape[1]
    emb_df = pd.DataFrame(emb, columns=[f'embedding_{i}' for i in range(n_comp)])
    emb_df.insert(0, 'row_id', list(df.index))
    emb_df.insert(1, 'partition', part)
    rows.append(emb_df)
emb_all = pd.concat(rows, ignore_index=True)
mlca_runtime.save_row_outputs(emb_all, 'embeddings.parquet')

print(f'Feature columns: {len(feature_cols)}')
print('Validation MSE: ' + ', '.join(f"{c['name']}={c['val_mse']:.6f}" for c in candidates))
print(f'Selected: {sel_name}')
print(f'Test {primary_metric}: {test_mse:.6f}')
print(f'Embedding dimension: {emb_all.shape[1] - 2}')
