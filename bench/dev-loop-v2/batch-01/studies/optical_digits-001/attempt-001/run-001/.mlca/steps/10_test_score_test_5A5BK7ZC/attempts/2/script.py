import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
primary = protocol['task']['primary_metric']
assert primary == 'reconstruction_error', primary

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
test = pd.read_parquet(inputs['test'])


def load(k):
    p = inputs[k]
    with open(p, 'rb') as f:
        try:
            return pickle.load(f)
        except Exception:
            import cloudpickle
            f.seek(0)
            return cloudpickle.load(f)


def flatten(m):
    stack = list(m.steps) if isinstance(m, Pipeline) else [('est', m)]
    out = []
    while stack:
        name, obj = stack.pop(0)
        if isinstance(obj, Pipeline):
            stack = list(obj.steps) + stack
        else:
            out.append((name, obj))
    seen, cleaned = set(), []
    for i, (name, obj) in enumerate(out):
        base, j = name, 1
        while name in seen:
            name = f'{base}_{j}'; j += 1
        seen.add(name); cleaned.append((name, obj))
    return cleaned


def build_front():
    pre = load('05_preprocess_fit_preprocessing_KTEB41J1/preprocessor.pkl')
    sel = load('06_features_retain_features_K0P2E6DF/selector_all.pkl')
    if isinstance(sel, Pipeline):
        return flatten(sel)
    return flatten(pre) + [('sel', sel)]


def ensure_pipeline(m):
    flat = flatten(m)
    last = flat[-1][1]
    if not isinstance(last, (PCA, MLPRegressor)):
        raise TypeError(f'Unsupported final estimator: {type(last).__name__}')
    if len(flat) >= 2:
        return Pipeline(flat)
    return Pipeline(build_front() + flat)


base = ensure_pipeline(load('07_model_train_baseline_Y4Q85156/model.pkl'))
alt = ensure_pipeline(load('08_model_train_alternative_CDFZSS39/alternative_model.pkl'))


def input_cols(pipe):
    for _, obj in pipe.steps:
        if hasattr(obj, 'feature_names_in_'):
            return list(obj.feature_names_in_)
    return None


def prep(pipe, df):
    cols = input_cols(pipe)
    X = df[cols].copy() if cols else df.copy()
    Z = np.asarray(Pipeline(pipe.steps[:-1]).transform(X), dtype=np.float64)
    return Z, pipe.steps[-1][1]


def mlp_bottleneck(mlp, Z):
    Ws, bs = mlp.coefs_, mlp.intercepts_
    widths = np.array([w.shape[1] for w in Ws[:-1]])
    mid = int(np.where(widths == widths.min())[0][0])
    A = np.asarray(Z, dtype=np.float64)
    for i in range(mid + 1):
        z = A @ Ws[i] + bs[i]
        a = mlp.activation
        if a == 'relu': A = np.maximum(z, 0.0)
        elif a == 'tanh': A = np.tanh(z)
        elif a == 'logistic': A = 1.0 / (1.0 + np.exp(-z))
        else: A = z
    return np.asarray(A, dtype=np.float64)


def recon(pipe, df):
    Z, est = prep(pipe, df)
    if isinstance(est, PCA):
        emb = np.asarray(est.transform(Z), dtype=np.float64)
        R = np.asarray(est.inverse_transform(emb), dtype=np.float64)
    elif isinstance(est, MLPRegressor):
        emb = mlp_bottleneck(est, Z)
        R = np.asarray(est.predict(Z), dtype=np.float64)
    else:
        raise TypeError(f'Unknown estimator: {type(est).__name__}')
    return float(np.mean((Z - R) ** 2)), emb, Z.shape[1]


def est_name(p):
    e = p.steps[-1][1]
    return {'PCA': 'PCA', 'MLPRegressor': 'MLPRegressor'}.get(type(e).__name__, type(e).__name__)


v_base, _, p_base = recon(base, val)
v_alt, _, p_alt = recon(alt, val)
if p_base != p_alt:
    raise RuntimeError(f'frontend feature dims differ: {p_base} vs {p_alt}')
selected = base if v_base <= v_alt else alt
sel_name = est_name(selected)

t_err, t_emb, _ = recon(selected, test)
_, tr_emb, _ = recon(selected, train)

tm = pd.DataFrame([{'model': sel_name, 'metric': primary, 'value': round(t_err, 6)}])
mlca_runtime.save_table(tm, 'test_metrics.csv', caption='Test reconstruction error')
mlca_runtime.record_metric(primary, round(t_err, 6), 'test')
mlca_runtime.record_fact_candidate('test_score', round(t_err, 6), source='test_metrics.csv')


def emb_frame(emb, df, part):
    out = pd.DataFrame(emb, columns=[f'embedding_{i}' for i in range(emb.shape[1])])
    out.insert(0, 'row_id', df.index.to_numpy())
    out.insert(1, 'partition', part)
    return out


all_emb = pd.concat([emb_frame(tr_emb, train, 'train'),
                     emb_frame(t_emb, test, 'test')], ignore_index=True)
mlca_runtime.save_row_outputs(all_emb, 'embeddings.parquet')

print(f'val base={v_base:.6f} alt={v_alt:.6f} selected={sel_name}')
print(f'test {primary}={t_err:.6f} emb_dim={tr_emb.shape[1]}')
print(f'rows={len(all_emb)}')
