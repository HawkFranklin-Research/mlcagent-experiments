import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs['_protocol'].read_text())
prim = proto['task']['primary_metric']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
test = pd.read_parquet(inputs['test'])
targets = proto.get('targets', []) or []
excluded = proto.get('excluded_columns', []) or []
gcol = proto.get('group_column', None)

def keylike(s):
    n = len(s)
    if s.isna().any(): return False
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.nunique() == n
    if pd.api.types.is_integer_dtype(s):
        return s.nunique() == n
    return False

feature_cols = [c for c in train.columns
                if c not in targets and c not in excluded and c != gcol and not keylike(train[c])]

with open(inputs['05_preprocess_fit_preprocessing_6DN7THMT/preprocessor.pkl'], 'rb') as f:
    preproc = pickle.load(f)
with open(inputs['06_features_retain_features_AKTAEF6F/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)
with open(inputs['07_model_train_baseline_V7ETXH3B/model.pkl'], 'rb') as f:
    baseline = pickle.load(f)
with open(inputs['08_model_train_alternative_G1KJG23J/alternative_model.pkl'], 'rb') as f:
    alt = pickle.load(f)

def frontend_for_bare(raw_cols):
    if hasattr(selector, 'feature_names_in_') and list(selector.feature_names_in_) == raw_cols:
        return selector
    return Pipeline([('preprocessor', preproc), ('selector', selector)])

def resolve(model, raw_cols):
    if isinstance(model, Pipeline) and len(model.steps) > 1:
        return Pipeline(model.steps[:-1]), model.steps[-1][1]
    if isinstance(model, Pipeline) and len(model.steps) == 1:
        return frontend_for_bare(raw_cols), model.steps[0][1]
    return frontend_for_bare(raw_cols), model

bf, bfinal = resolve(baseline, feature_cols)
af, afinal = resolve(alt, feature_cols)

def kind_of(e):
    if isinstance(e, PCA): return 'pca'
    if isinstance(e, MLPRegressor): return 'mlp'
    return 'pca' if not hasattr(e, 'predict') else 'mlp'

b_kind, a_kind = kind_of(bfinal), kind_of(afinal)

def recon_error(front, final, kind, X):
    F = front.transform(X)
    Fa = np.asarray(F, float)
    if kind == 'pca':
        rec = final.inverse_transform(final.transform(F))
    else:
        rec = final.predict(F)
    return float(np.mean((Fa - np.asarray(rec, float))**2))

b_val = recon_error(bf, bfinal, b_kind, val[feature_cols])
a_val = recon_error(af, afinal, a_kind, val[feature_cols])
print(f"features={len(feature_cols)} b={type(bfinal).__name__} a={type(afinal).__name__}")
print(f"val recon: baseline={b_val:.6f} alt={a_val:.6f}")

if b_val <= a_val:
    ch_front, ch_final, ch_kind, ch_name = bf, bfinal, b_kind, 'baseline'
else:
    ch_front, ch_final, ch_kind, ch_name = af, afinal, a_kind, 'alternative'

test_mse = recon_error(ch_front, ch_final, ch_kind, test[feature_cols])

dfm = pd.DataFrame({'model': [ch_name], 'metric': [prim],
                    'value': [float(f'{test_mse:.4g}')]})
mlca_runtime.save_table(dfm, 'test_metrics.csv')
mlca_runtime.record_metric(prim, test_mse, 'test')
mlca_runtime.record_fact_candidate('test_score', test_mse, source='test_metrics.csv')

def activation(name, x):
    if name == 'relu': return np.maximum(0, x)
    if name == 'tanh': return np.tanh(x)
    if name == 'logistic': return 1.0 / (1.0 + np.exp(-x))
    if name == 'identity': return x
    raise ValueError('unknown activation ' + str(name))

def bottleneck(mlp, F):
    sizes = mlp.hidden_layer_sizes
    bi = int(np.argmin(sizes))
    h = np.asarray(F, float)
    for i in range(bi + 1):
        h = h @ mlp.coefs_[i] + mlp.intercepts_[i]
        h = activation(mlp.activation, h)
    return np.asarray(h, float)

def embed(front, final, kind, X):
    F = front.transform(X)
    if kind == 'pca':
        return np.asarray(final.transform(F), float)
    return bottleneck(final, F)

Etr = embed(ch_front, ch_final, ch_kind, train[feature_cols])
Ete = embed(ch_front, ch_final, ch_kind, test[feature_cols])

def pack(E, ids, part):
    df = pd.DataFrame(E, columns=[f'embedding_{i}' for i in range(E.shape[1])])
    df.insert(0, 'partition', part)
    df.insert(0, 'row_id', ids)
    return df

emb = pd.concat([pack(Etr, train.index.to_numpy(), 'train'),
                 pack(Ete, test.index.to_numpy(), 'test')], ignore_index=True)
mlca_runtime.save_row_outputs(emb, 'embeddings.parquet')
print(f"embedding_dim={Etr.shape[1]} model={ch_name} test_{prim}={test_mse:.6f}")
