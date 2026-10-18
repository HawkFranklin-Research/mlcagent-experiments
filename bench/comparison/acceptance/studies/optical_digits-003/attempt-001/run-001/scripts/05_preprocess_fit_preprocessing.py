import mlca_runtime, json, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
protocol = json.loads(inp['_protocol'].read_text())

def pget(key, default):
    if key in protocol: return protocol[key]
    t = protocol.get('task')
    if isinstance(t, dict) and key in t: return t[key]
    return default

targets = pget('targets', []) or []
excluded = pget('excluded_columns', []) or []
group = pget('group_column', None)

drop = set(map(str, targets)) | set(map(str, excluded))
if group: drop.add(str(group))

def id_like(name, s):
    if name.lower() in ('id','index','row_id','unnamed: 0'): return True
    if s.dtype.kind in 'OUS':
        return s.notna().all() and s.nunique(dropna=True) == len(s)
    if s.dtype.kind in 'iu':
        s2 = s.dropna()
        if len(s2) == len(s) and s2.nunique() == len(s2):
            v = np.sort(s2.values)
            if len(v) > 1 and (v[-1] - v[0]) == len(v) - 1:
                return True
    return False

features = [c for c in train.columns if c not in drop and not id_like(c, train[c])]

obj_cols = [c for c in features if train[c].dtype.kind in 'OUS']
n_ph = sum(1 for c in obj_cols if train[c].astype(str).str.strip().isin(['?','-','NA','N/A','']).any())

numeric_cols, cat_cols = [], []
for c in features:
    s = train[c]
    if s.dtype.kind in 'fiu' or s.dtype.kind == 'b':
        numeric_cols.append(c)
    elif s.dtype.kind in 'OUS':
        is_date = False
        try:
            sm = s.dropna().head(50)
            if len(sm) >= 5:
                pd.to_datetime(sm, errors='raise'); is_date = True
        except Exception:
            is_date = False
        if not is_date and s.nunique(dropna=True) <= max(2, int(0.05 * len(train))):
            cat_cols.append(c)

log_cols, plain_num = [], []
for c in numeric_cols:
    s = train[c].dropna()
    if len(s) > 2 and (s >= 0).all() and float(pd.Series(s).skew()) > 1.0:
        log_cols.append(c)
    else:
        plain_num.append(c)

def num_pipe(use_log):
    steps = []
    if use_log:
        steps.append(('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one', validate=False)))
    steps += [('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]
    return Pipeline(steps)

transformers = []
if log_cols: transformers.append(('num_log', num_pipe(True), log_cols))
if plain_num: transformers.append(('num', num_pipe(False), plain_num))
if cat_cols: transformers.append(('cat', Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
]), cat_cols))

ct = ColumnTransformer(transformers, remainder='drop')
Xtr = train[features].copy()
ct.set_output(transform='pandas')
fitted = mlca_runtime.fit_record(ct, Xtr, partition='train')
width = fitted.transform(Xtr.head(5)).shape[1]
mlca_runtime.save_model(fitted, 'preprocessor.pkl')
mlca_runtime.record_metric('n_features', float(len(features)), 'train')
print(f"placeholders_cols={n_ph}")
print(f"features={len(features)} numeric={len(numeric_cols)} log1p={len(log_cols)} cat={len(cat_cols)} out_width={width}")
