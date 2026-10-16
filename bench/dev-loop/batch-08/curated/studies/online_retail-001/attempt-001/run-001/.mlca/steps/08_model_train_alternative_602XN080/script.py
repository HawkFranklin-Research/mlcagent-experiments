import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])

def load_pkl(key):
    p = inputs.get(key)
    return pickle.loads(Path(p).read_bytes()) if p else None

pre = load_pkl("05_preprocess_fit_preprocessing_FN0Z178K/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_YJWZ54M0/selector_all.pkl")
base = load_pkl("07_model_train_baseline_SMYBENGX/model.pkl")

if isinstance(base, Pipeline) and len(base.steps) > 1:
    frontend = base[:-1]
elif isinstance(sel, Pipeline):
    frontend = sel
elif pre is not None and sel is not None:
    frontend = Pipeline([('pre', pre), ('sel', sel)])
elif pre is not None:
    frontend = pre
elif sel is not None:
    frontend = sel
else:
    frontend = 'passthrough'

def get_cols(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return None

cols = get_cols(frontend)
if cols is None:
    targets = set(proto.get('targets', []))
    excluded = set(proto.get('excluded_columns', []))
    group = proto.get('group_column')
    def key_like(s):
        if pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            return s.nunique(dropna=False) == len(s)
        if pd.api.types.is_integer_dtype(s):
            return s.nunique(dropna=False) == len(s)
        return False
    cols = [c for c in train.columns if c not in targets and c not in excluded and c != group and not key_like(train[c])]

missing = [c for c in cols if c not in train.columns]
if missing:
    raise KeyError(f"MLCA: missing feature columns {missing}; available {list(train.columns)}")
X = train.loc[:, cols].copy()

ks = list(range(2, 9))
rng = np.random.RandomState(42)
rows = []
for k in ks:
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10, batch_size=1024, max_iter=100)
    pipe = Pipeline([('features', frontend), ('kmeans', km)])
    fitted = mlca_runtime.fit_record(pipe, X, "train")
    labels = fitted.named_steps['kmeans'].labels_
    ft = fitted.named_steps['features']
    Xt = X if ft == 'passthrough' else ft.transform(X)
    n = Xt.shape[0]
    if n > 10000:
        idx = rng.choice(n, 10000, replace=False)
        Xs, ls = Xt[idx], labels[idx]
    else:
        Xs, ls = Xt, labels
    Xd = Xs.toarray() if hasattr(Xs, 'toarray') else Xs
    sil = silhouette_score(Xs, ls, sample_size=min(10000, len(ls)), random_state=42)
    db = davies_bouldin_score(Xd, ls)
    ch = calinski_harabasz_score(Xd, ls)
    aris = []
    for b in range(3):
        w = rng.poisson(1.0, size=n)
        km_b = MiniBatchKMeans(n_clusters=k, random_state=42+b, n_init=10, batch_size=1024, max_iter=100)
        pipe_b = Pipeline([('features', frontend), ('kmeans', km_b)])
        fit_b = mlca_runtime.fit_record(pipe_b, X, "train", kmeans__sample_weight=w)
        aris.append(adjusted_rand_score(labels, fit_b.named_steps['kmeans'].labels_))
    rows.append({'k': k, 'silhouette': sil, 'davies_bouldin': db, 'calinski_harabasz': ch, 'stability': float(np.mean(aris))})

res = pd.DataFrame(rows)
def norm(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-12)
score = norm(res['silhouette']) + (1 - norm(res['davies_bouldin'])) + norm(res['calinski_harabasz']) + norm(res['stability'])
best = res.iloc[int(score.argmax())]
best_k = int(best['k'])

final_km = MiniBatchKMeans(n_clusters=best_k, random_state=42, n_init=10, batch_size=1024, max_iter=100)
final_pipe = Pipeline([('features', frontend), ('kmeans', final_km)])
final_model = mlca_runtime.fit_record(final_pipe, X, "train")
mlca_runtime.save_model(final_model, "alternative_model.pkl")
mlca_runtime.record_metric("silhouette", float(best['silhouette']), split="train")
mlca_runtime.record_metric("davies_bouldin", float(best['davies_bouldin']), split="train")
mlca_runtime.record_metric("calinski_harabasz", float(best['calinski_harabasz']), split="train")
mlca_runtime.record_metric("stability", float(best['stability']), split="train")
mlca_runtime.record_metric("selected_k", float(best_k), split="train")
print("done", best_k, X.shape)
