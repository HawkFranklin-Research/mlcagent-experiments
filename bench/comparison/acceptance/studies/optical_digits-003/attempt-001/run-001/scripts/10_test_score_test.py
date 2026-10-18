import pickle
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
task = protocol['task']
primary_metric = task['primary_metric']
targets = protocol.get('targets', [])
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
test = pd.read_parquet(inputs['test'])

excl = set(targets) | set(excluded)
if group_col:
    excl.add(group_col)
feature_cols = []
for c in train.columns:
    if c in excl:
        continue
    col = train[c]
    if col.dtype == object or pd.api.types.is_string_dtype(col):
        if col.nunique() == len(col):
            continue
    if pd.api.types.is_integer_dtype(col):
        if col.nunique() == len(col):
            continue
    feature_cols.append(c)

with open(inputs['05_preprocess_fit_preprocessing_JYNCWEGN/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_retain_features_HY561TZJ/selector_all.pkl'], 'rb') as f:
    sel = pickle.load(f)

def frontend_steps(pre, sel, X):
    if sel is not None and isinstance(sel, Pipeline):
        if hasattr(sel, 'feature_names_in_'):
            try:
                if list(sel.feature_names_in_) == list(X.columns):
                    return [sel]
            except Exception:
                pass
        if pre is not None:
            for step in sel.named_steps.values():
                if step is pre:
                    return [sel]
    steps = []
    if pre is not None:
        steps.append(pre)
    if sel is not None:
        steps.append(sel)
    return steps

def split_pipeline(obj):
    if isinstance(obj, Pipeline) and len(obj.steps) > 0:
        last = obj.steps[-1][1]
        if isinstance(last, (PCA, MLPRegressor)):
            frontend = Pipeline(obj.steps[:-1]) if len(obj.steps) > 1 else None
            return frontend, last
    return None, obj

def mlp_embedding(mlp, X):
    coefs = mlp.coefs_
    intercepts = mlp.intercepts_
    act = mlp.activation
    n_hidden = len(coefs) - 1
    mid = (n_hidden - 1) // 2 if n_hidden > 0 else 0
    A = np.asarray(X, dtype=float)
    for i in range(mid + 1):
        A = A @ coefs[i] + intercepts[i]
        if i < n_hidden:
            if act == 'relu':
                A = np.maximum(A, 0)
            elif act == 'tanh':
                A = np.tanh(A)
            elif act == 'logistic':
                A = 1.0 / (1.0 + np.exp(-A))
    return A

candidates = [
    ('07_baseline', inputs['07_model_train_baseline_V6BDRB4H/model.pkl']),
    ('08_alternative', inputs['08_model_train_alternative_WZMYQEJM/alternative_model.pkl'])
]

results = []
for name, path in candidates:
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    fe_pipe, final = split_pipeline(obj)
    if fe_pipe is not None:
        def frontend(D, p=fe_pipe):
            return p.transform(D)
    else:
        steps = frontend_steps(pre, sel, train[feature_cols])
        def frontend(D, steps=steps):
            Z = D
            for s in steps:
                if s is not None:
                    Z = s.transform(Z)
            return Z

    Xf_val = frontend(val[feature_cols])
    if isinstance(final, PCA):
        Xrec = final.inverse_transform(final.transform(Xf_val))
        model_type = 'PCA'
    else:
        Xrec = final.predict(Xf_val)
        model_type = 'MLP'
    mse = float(np.mean((Xf_val - Xrec) ** 2))
    results.append((name, model_type, final, frontend, mse))

best_val = min(results, key=lambda r: r[4])
best_name, best_type, best_final, best_frontend, best_val_mse = best_val

Xf_test = best_frontend(test[feature_cols])
if isinstance(best_final, PCA):
    Xrec_test = best_final.inverse_transform(best_final.transform(Xf_test))
else:
    Xrec_test = best_final.predict(Xf_test)
test_mse = float(np.mean((Xf_test - Xrec_test) ** 2))

metrics_df = pd.DataFrame([{'model': best_name, 'metric': primary_metric, 'value': test_mse}])
mlca_runtime.save_table(metrics_df, 'test_metrics.csv', caption='Test reconstruction error')
mlca_runtime.record_metric(primary_metric, test_mse, split='test')
mlca_runtime.record_fact_candidate('test_score', test_mse, source='test_metrics.csv')

emb_parts = []
for part, df in [('train', train), ('test', test)]:
    Xf = best_frontend(df[feature_cols])
    if isinstance(best_final, PCA):
        emb = best_final.transform(Xf)
    else:
        emb = mlp_embedding(best_final, Xf)
    emb_df = pd.DataFrame(emb, columns=[f'embedding_{i}' for i in range(emb.shape[1])])
    emb_df.insert(0, 'row_id', df.index)
    emb_df.insert(1, 'partition', part)
    emb_parts.append(emb_df)

emb_all = pd.concat(emb_parts, ignore_index=False)
mlca_runtime.save_row_outputs(emb_all, 'embeddings.parquet')

print(f"selected {best_name} ({best_type}) val_mse={best_val_mse:.6f}")
print(f"test_mse={test_mse:.6f}, embeddings_shape={emb_all.shape}")
