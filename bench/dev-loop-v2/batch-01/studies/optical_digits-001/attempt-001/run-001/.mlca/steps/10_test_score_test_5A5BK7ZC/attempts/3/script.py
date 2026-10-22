import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
primary = json.loads(inp['_protocol'].read_text())['task']['primary_metric']
train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
test = pd.read_parquet(inp['test'])


def load(k):
    with open(inp[k], 'rb') as f:
        try:
            return pickle.load(f)
        except Exception:
            import cloudpickle
            f.seek(0)
            return cloudpickle.load(f)


def unwrap(m):
    if isinstance(m, dict):
        for k in ('model', 'pipeline', 'estimator', 'final_estimator', 'final'):
            if k in m:
                return m[k]
    if isinstance(m, tuple) and len(m):
        return m[0]
    if hasattr(m, 'best_estimator_'):
        return m.best_estimator_
    return m


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
    for name, obj in out:
        base, j = name, 1
        while name in seen:
            name = f'{base}_{j}'; j += 1
        seen.add(name); cleaned.append((name, obj))
    return cleaned


def build_front():
    pre = load('05_preprocess_fit_preprocessing_KTEB41J1/preprocessor.pkl')
    sel = load('06_features_retain_features_K0P2E6DF/selector_all.pkl')
    return flatten(sel) if isinstance(sel, Pipeline) else flatten(pre) + [('sel', sel)]


def is_encoder(e): return hasattr(e, 'transform') and hasattr(e, 'inverse_transform')
def is_autoenc(e): return hasattr(e, 'predict') and hasattr(e, 'coefs_') and hasattr(e, 'intercepts_')


def ensure_pipeline(m):
    flat = flatten(unwrap(m))
    last = flat[-1][1]
    if not (is_encoder(last) or is_autoenc(last)):
        print(f'loaded final: {type(last).__module__}.{type(last).__name__}')
        raise TypeError(f'Unsupported final: {type(last).__name__}')
    return Pipeline(flat) if len(flat) >= 2 else Pipeline(build_front() + flat)


base = ensure_pipeline(load('07_model_train_baseline_Y4Q85156/model.pkl'))
alt = ensure_pipeline(load('08_model_train_alternative_CDFZSS39/alternative_model.pkl'))


def prep(pipe, df):
    head = Pipeline(pipe.steps[:-1])
    cols = None
    for _, obj in head.steps:
        if hasattr(obj, 'feature_names_in_'):
            cols = list(obj.feature_names_in_); break
    X = df[cols].copy() if cols else df.copy()
    return np.asarray(head.transform(X), dtype=np.float64), pipe.steps[-1][1]


def bottleneck(mlp, Z):
    Ws, bs = mlp.coefs_, mlp.intercepts_
    widths = np.array([w.shape[1] for w in Ws[:-1]])
    mid = int(np.where(widths == widths.min())[0][0])
    A = np.asarray(Z, dtype=np.float64)
    for i in range(mid + 1):
        z = A @ Ws[i] + bs[i]
        a = mlp.activation
        if a == 'relu': A = np.maximum(z, 0.0)
        elif a == 'tanh': A = np.tanh(z)
        elif a == 'logistic': A = expit(z)
        else: A = z
    return np.asarray(A, dtype=np.float64)


def recon(pipe, df):
    Z, est = prep(pipe, df)
    if is_encoder(est):
        emb = np.asarray(est.transform(Z), dtype=np.float64)
        R = np.asarray(est.inverse_transform(emb), dtype=np.float64)
    else:
        emb = bottleneck(est, Z)
        R = np.asarray(est.predict(Z), dtype=np.float64)
    err = float(np.mean((Z - R) ** 2))
    if primary in ('rmse', 'root_mean_squared_error'):
        err = float(np.sqrt(err))
    return err, emb, Z.shape[1]


vb, _, pb = recon(base, val)
va, _, pa = recon(alt, val)
if pb != pa:
    raise RuntimeError(f'frontend dims differ: {pb} vs {pa}')
selected = base if vb <= va else alt
t_err, t_emb, _ = recon(selected, test)
_, tr_emb, _ = recon(selected, train)

tm = pd.DataFrame([{'model': type(selected.steps[-1][1]).__name__,
                    'metric': primary, 'value': round(t_err, 6)}])
mlca_runtime.save_table(tm, 'test_metrics.csv', caption='Test reconstruction error')
mlca_runtime.record_metric(primary, round(t_err, 6), 'test')
mlca_runtime.record_fact_candidate('test_score', round(t_err, 6), source='test_metrics.csv')


def ef(emb, df, part):
    o = pd.DataFrame(emb, columns=[f'embedding_{i}' for i in range(emb.shape[1])])
    o.insert(0, 'row_id', df.index.to_numpy())
    o.insert(1, 'partition', part)
    return o


all_emb = pd.concat([ef(tr_emb, train, 'train'), ef(t_emb, test, 'test')], ignore_index=True)
mlca_runtime.save_row_outputs(all_emb, 'embeddings.parquet')

print(f'val base={vb:.6f} alt={va:.6f}')
print(f'test {primary}={t_err:.6f} emb_dim={tr_emb.shape[1]}')
print(f'rows={len(all_emb)}')
