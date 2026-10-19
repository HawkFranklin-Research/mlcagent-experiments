import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
primary_metric = proto['task']['primary_metric']
targets = proto.get('targets', [])
excluded = proto.get('excluded_columns', [])
group = proto.get('group_column', None)

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
test = pd.read_parquet(inp['test'])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
features = [c for c in train.columns if c not in drop]
features = [c for c in features if not (train[c].dtype == object and train[c].nunique(dropna=False) == len(train))]

pre = pickle.load(open(inp['05_preprocess_fit_preprocessing_4ZZYYY8Y/preprocessor.pkl'], 'rb'))
sel = pickle.load(open(inp['06_features_retain_features_E2FAAE9F/selector_all.pkl'], 'rb'))

def input_names(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if hasattr(obj, 'named_steps') and obj.named_steps:
        return input_names(list(obj.named_steps.values())[0])
    return None

sel_names = input_names(sel)
if sel_names is not None and set(sel_names) == set(features):
    frontend = sel
else:
    pre_names = input_names(pre)
    if pre_names is not None and set(pre_names) != set(features):
        raise ValueError(f"Preprocessor expects {pre_names}, features {features}")
    frontend = Pipeline([('pre', pre), ('sel', sel)])

cols = getattr(frontend, 'feature_names_in_', features)
def transform(front, X):
    out = front.transform(X)
    if hasattr(out, 'toarray'):
        out = out.toarray()
    return np.asarray(out)

Xtr = transform(frontend, train[list(cols)])
Xva = transform(frontend, val[list(cols)])
Xte = transform(frontend, test[list(cols)])

def final_est(obj):
    if hasattr(obj, 'named_steps') and obj.named_steps:
        return list(obj.named_steps.values())[-1]
    return obj

cands = {
    'baseline': final_est(pickle.load(open(inp['07_model_train_baseline_Q9V0XT19/model.pkl'], 'rb'))),
    'alternative': final_est(pickle.load(open(inp['08_model_train_alternative_GWTD8WJA/alternative_model.pkl'], 'rb'))),
}

def recon_mse(est, X):
    if isinstance(est, PCA):
        Xrec = est.inverse_transform(est.transform(X))
    elif isinstance(est, MLPRegressor):
        Xrec = est.predict(X)
    else:
        raise TypeError(type(est))
    return float(np.mean((X - Xrec) ** 2))

val_scores = {}
for name, est in cands.items():
    if hasattr(est, 'n_features_in_') and est.n_features_in_ != Xtr.shape[1]:
        raise ValueError(f"{name} expects {est.n_features_in_}, got {Xtr.shape[1]}")
    val_scores[name] = recon_mse(est, Xva)
    mlca_runtime.record_metric(f"val_{primary_metric}_{name}", val_scores[name], split='val')

best_name = min(val_scores, key=val_scores.get)
best_est = cands[best_name]
test_score = recon_mse(best_est, Xte)

mlca_runtime.record_metric(primary_metric, test_score, split='test')
mlca_runtime.record_fact_candidate('test_score', test_score, source='test_metrics.csv')

metrics_df = pd.DataFrame([{'model': best_name, 'metric': primary_metric, 'value': float(f"{test_score:.4g}")}])
mlca_runtime.save_table(metrics_df, "test_metrics.csv")

def activation(z, name):
    if name == 'relu': return np.maximum(z, 0)
    if name == 'tanh': return np.tanh(z)
    if name == 'logistic': return 1.0 / (1.0 + np.exp(-z))
    if name == 'identity': return z
    raise ValueError(name)

def embedding(est, X):
    if isinstance(est, PCA):
        return est.transform(X)
    if isinstance(est, MLPRegressor):
        h = X
        widths = est.hidden_layer_sizes
        bidx = int(np.argmin(widths))
        for i in range(bidx + 1):
            h = activation(h @ est.coefs_[i] + est.intercepts_[i], est.activation)
        return h
    raise TypeError(type(est))

def emb_df(X_raw, X_front, part):
    E = embedding(best_est, X_front)
    if not np.all(np.isfinite(E)):
        raise ValueError("Non-finite embedding")
    df = pd.DataFrame(E, columns=[f'embedding_{i}' for i in range(E.shape[1])])
    df.insert(0, 'row_id', X_raw.index)
    df.insert(1, 'partition', part)
    return df

emb = pd.concat([emb_df(train[list(cols)], Xtr, 'train'), emb_df(test[list(cols)], Xte, 'test')], ignore_index=True)
if emb['row_id'].duplicated().any():
    raise ValueError("row_id must be unique across train and test")
mlca_runtime.save_row_outputs(emb, "embeddings.parquet")
mlca_runtime.record_metric("embedding_dim", int(emb.shape[1] - 2), split='train')

print(f"train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}")
print(f"val {val_scores}")
print(f"selected {best_name}, test {primary_metric}={test_score:.4f}")
