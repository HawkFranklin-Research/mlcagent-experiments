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
    try:
        with open(p, 'rb') as f:
            return pickle.load(f)
    except Exception:
        import cloudpickle
        with open(p, 'rb') as f:
            return cloudpickle.load(f)


def ensure_pipeline(m):
    if isinstance(m, Pipeline) and len(m.steps) >= 2:
        return m
    pre = load('05_preprocess_fit_preprocessing_KTEB41J1/preprocessor.pkl')
    sel = load('06_features_retain_features_K0P2E6DF/selector_all.pkl')
    if isinstance(sel, Pipeline) and len(sel.steps) >= 1:
        return Pipeline([('front', sel), ('est', m)])
    return Pipeline([('pre', pre), ('sel', sel), ('est', m)])


base = ensure_pipeline(load('07_model_train_baseline_Y4Q85156/model.pkl'))
alt = ensure_pipeline(load('08_model_train_alternative_CDFZSS39/alternative_model.pkl'))


def get_input_cols(pipe):
    if isinstance(pipe, Pipeline) and len(pipe.steps) >= 1:
        first = pipe.steps[0][1]
        if hasattr(first, 'feature_names_in_'):
            return list(first.feature_names_in_)
    if hasattr(pipe, 'feature_names_in_'):
        return list(pipe.feature_names_in_)
    return None


def prep(pipe, df):
    cols = get_input_cols(pipe)
    X = df[cols].copy() if cols else df.copy()
    if isinstance(pipe, Pipeline) and len(pipe.steps) > 1:
        Z = np.asarray(Pipeline(pipe.steps[:-1]).transform(X), dtype=np.float64)
        est = pipe.steps[-1][1]
    else:
        Z = np.asarray(X.values, dtype=np.float64)
        est = pipe
    return Z, est


def mlp_bottleneck(mlp, Z):
    Ws, bs = mlp.coefs_, mlp.intercepts_
    widths = np.array([w.shape[1] for w in Ws[:-1]])
    mins = np.where(widths == widths.min())[0]
    mid = int(mins[len(mins) // 2])
    act = mlp.activation
    A = Z
    for i in range(mid + 1):
        z = A @ Ws[i] + bs[i]
        if act == 'relu':
            A = np.maximum(z, 0.0)
        elif act == 'tanh':
            A = np.tanh(z)
        elif act == 'logistic':
            A = 1.0 / (1.0 + np.exp(-z))
        else:
            A = z
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
        raise TypeError(f'Unknown estimator: {type(est)}')
    return float(np.mean((Z - R) ** 2)), emb


def est_name(p):
    e = p.steps[-1][1] if isinstance(p, Pipeline) else p
    if isinstance(e, PCA):
        return 'PCA'
    if isinstance(e, MLPRegressor):
        return 'MLPRegressor'
    return type(e).__name__


v_base, _ = recon(base, val)
v_alt, _ = recon(alt, val)
selected = base if v_base <= v_alt else alt
sel_name = est_name(selected)

t_err, t_emb = recon(selected, test)
_, tr_emb = recon(selected, train)

tm = pd.DataFrame([{'model': sel_name, 'metric': primary, 'value': round(t_err, 6)}])
mlca_runtime.save_table(tm, 'test_metrics.csv',
                        caption='Test reconstruction error of validation-selected candidate')
mlca_runtime.record_metric(primary, round(t_err, 6), 'test')
mlca_runtime.record_fact_candidate('test_score', round(t_err, 6), source='test_metrics.csv')


def emb_df(emb, df, partition):
    out = pd.DataFrame(emb, columns=[f'embedding_{i}' for i in range(emb.shape[1])])
    out.insert(0, 'row_id', df.index.to_numpy())
    out.insert(1, 'partition', partition)
    return out


all_emb = pd.concat([emb_df(tr_emb, train, 'train'), emb_df(t_emb, test, 'test')],
                    ignore_index=True)
mlca_runtime.save_row_outputs(all_emb, 'embeddings.parquet')

print(f'val_base={v_base:.6f} val_alt={v_alt:.6f} selected={sel_name}')
print(f'test_{primary}={t_err:.6f} emb_dim={tr_emb.shape[1]}')
print(f'embeddings_rows={len(all_emb)}')
