import json
from pathlib import Path
import numpy as np
import pandas as pd
import mlca_runtime as rt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

rt.set_seed(42); SEED = 42
inp = rt.inputs()
proto = json.loads(inp['_protocol'].read_text())
targets = list(proto.get('targets') or proto.get('task', {}).get('targets') or [])
excluded = list(proto.get('excluded_columns') or [])
group = proto.get('group_column')
train = pd.read_parquet(inp['train'])
drop = set(targets) | set(excluded) | ({str(group)} if group else set())

def is_key(s):
    if len(s) < 2 or s.isna().any(): return False
    if pd.api.types.is_integer_dtype(s):
        return s.nunique() == len(s) and int(s.max()) - int(s.min()) + 1 == len(s)
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique() == len(s)
    return False

def looks_dt(s):
    v = s.dropna()
    if len(v) == 0: return False
    try:
        pd.to_datetime(v.head(20), errors='raise', format='mixed'); return True
    except Exception:
        return False

feat = [c for c in train.columns if c not in drop]
feat = [c for c in feat if not is_key(train[c])]
num, cat, date = [], [], []
for c in feat:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s) or (pd.api.types.is_object_dtype(s) and looks_dt(s)):
        date.append(c)
    elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num.append(c)
    else:
        cat.append(c)
n = len(train)
thr = max(5, int(0.05 * n))
low = [c for c in cat if train[c].nunique() <= min(50, thr)]
high = [c for c in cat if c not in low]
feat = num + low
if not feat or n < 3:
    raise RuntimeError(f'no usable features; cols={list(train.columns)[:20]}')

def build(k):
    trs = []
    if num:
        trs.append(('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                                     ('sc', StandardScaler())]), num))
    if low:
        trs.append(('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                                     ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), low))
    pre = ColumnTransformer(trs, remainder='drop'); pre.set_output(transform='pandas')
    bs = min(1024, max(256, n // 10))
    return Pipeline([('pre', pre), ('km', MiniBatchKMeans(n_clusters=k, random_state=SEED,
                                                           n_init=3, batch_size=bs, max_iter=300))])

X = train[feat]
Xval = None
if 'val' in inp:
    try:
        vt = pd.read_parquet(inp['val'])
        if all(c in vt.columns for c in feat): Xval = vt[feat]
    except Exception:
        Xval = None

ks = sorted(set(range(2, 11)))
cp = inp.get('07_model_train_baseline_RA5A8N63/kmeans_candidates.csv')
if cp is not None and Path(cp).exists():
    try:
        cand = pd.read_csv(cp)
        col = 'k' if 'k' in cand.columns else ('n_clusters' if 'n_clusters' in cand.columns else None)
        if col: ks = sorted({int(v) for v in cand[col].dropna().unique()})
    except Exception:
        pass
ks = [k for k in ks if 2 <= k < n][:12] or [2]

ss = min(10000, n); rng = np.random.default_rng(SEED); rows = []
for k in ks:
    fit = rt.fit_record(build(k), X, 'train')
    Xt = fit.named_steps['pre'].transform(X)
    lab = fit.named_steps['km'].predict(Xt)
    multi = len(set(lab)) > 1
    sil = float(silhouette_score(Xt, lab, sample_size=ss, random_state=SEED)) if multi else -1.0
    db = float(davies_bouldin_score(Xt, lab)) if multi else np.nan
    ch = float(calinski_harabasz_score(Xt, lab)) if multi else np.nan
    w = rng.poisson(1.0, size=n).astype(float)
    wb = rt.fit_record(build(k), X, 'train', km__sample_weight=w)
    lab_b = wb.named_steps['km'].predict(wb.named_steps['pre'].transform(X))
    ari = float(adjusted_rand_score(lab, lab_b))
    rows.append({'k': k, 'silhouette': sil, 'davies_bouldin': db,
                 'calinski_harabasz': ch, 'stability_ari': ari})

res = pd.DataFrame(rows)
best = res.sort_values(['silhouette', 'stability_ari'], ascending=False).iloc[0]
K = int(best['k'])
final = rt.fit_record(build(K), X, 'train')
rt.save_model(final, 'alternative_model.pkl')
rt.record_metric('silhouette', float(best['silhouette']), 'train')
rt.record_metric('stability_ari', float(best['stability_ari']), 'train')
rt.record_metric('n_clusters', float(K), 'train')
rt.record_metric('n_features', float(len(feat)), 'train')
if Xval is not None:
    try:
        Xv = final.named_steps['pre'].transform(Xval)
        lv = final.named_steps['km'].predict(Xv)
        if len(set(lv)) > 1 and len(Xv) > 1:
            rt.record_metric('silhouette', float(silhouette_score(Xv, lv, sample_size=min(10000, len(Xv)), random_state=SEED)), 'val')
    except Exception:
        pass
print(f'rows={n} feats={len(feat)} num={len(num)} cat={len(low)} dropped_key_or_high={len(high) + len(date)}')
print(f'k_candidates={ks}')
print(f'chosen_k={K} silhouette={float(best["silhouette"]):.4f} ari={float(best["stability_ari"]):.4f}')
