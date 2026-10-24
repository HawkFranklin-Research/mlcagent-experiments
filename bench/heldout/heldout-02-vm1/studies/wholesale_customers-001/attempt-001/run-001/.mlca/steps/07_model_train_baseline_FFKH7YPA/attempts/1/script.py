import mlca_runtime, pickle, json, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs['train'])
protocol = json.loads(inputs['_protocol'].read_text())
pre = pickle.load(open(inputs['05_preprocess_fit_preprocessing_RXV573Z5/preprocessor.pkl'], 'rb'))
sel = pickle.load(open(inputs['06_features_retain_features_6C317WGV/selector_all.pkl'], 'rb'))

# Determine raw feature columns from saved preprocessor if available
if hasattr(pre, 'feature_names_in_'):
    feature_cols = list(pre.feature_names_in_)
elif hasattr(sel, 'feature_names_in_'):
    feature_cols = list(sel.feature_names_in_)
else:
    drop = set(protocol.get('excluded_columns', [])) | set(protocol.get('targets', []))
    g = protocol.get('group_column')
    if g: drop.add(g)
    feature_cols = []
    for c in train.columns:
        if c in drop: continue
        s = train[c]
        if s.dtype == 'object' or str(s.dtype).startswith('string'):
            if s.nunique() == len(train): continue
        elif pd.api.types.is_integer_dtype(s):
            if s.nunique() == len(train) and (s.max() - s.min() == len(train) - 1): continue
        feature_cols.append(c)

X = train[feature_cols].copy()

# Decide frontend: selector may already wrap preprocessing
sel_steps = sel.named_steps if isinstance(sel, Pipeline) else {}
has_pre = any(isinstance(s, (ColumnTransformer, Pipeline)) for s in sel_steps.values()) if sel_steps else False
steps = [('sel', sel)] if has_pre else [('pre', pre), ('sel', sel)]

best_k, best_score, best_model = None, -1, None
for k in range(2, 9):
    pipe = Pipeline(steps + [('kmeans', KMeans(n_clusters=k, n_init=10, random_state=42))])
    fitted = mlca_runtime.fit_record(pipe, X, 'train')
    Xt = fitted[:-1].transform(X)
    labels = fitted.named_steps['kmeans'].labels_
    n = len(Xt)
    if n > 10000:
        idx = np.random.RandomState(42).choice(n, 10000, replace=False)
        Xs = Xt.iloc[idx] if hasattr(Xt, 'iloc') else Xt[idx]
        ls = labels[idx]
    else:
        Xs, ls = Xt, labels
    if len(np.unique(ls)) < 2:
        continue
    sil = silhouette_score(Xs, ls, random_state=42)
    if sil > best_score:
        best_score, best_k, best_model = sil, k, fitted

mlca_runtime.save_model(best_model, 'model.pkl')
mlca_runtime.record_metric('silhouette', float(best_score), 'train')
mlca_runtime.record_metric('n_clusters', float(best_k), 'train')
print(f'k={best_k} silhouette={best_score:.4f} n={len(X)}')
