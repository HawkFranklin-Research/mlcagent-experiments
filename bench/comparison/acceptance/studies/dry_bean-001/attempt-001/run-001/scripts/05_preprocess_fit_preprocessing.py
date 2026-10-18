import json, numpy as np, pandas as pd, mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
p = mlca_runtime.inputs()
proto = json.loads(p['_protocol'].read_text())
for k in ('targets', 'excluded_columns', 'task'):
    if k not in proto:
        raise KeyError(f"Missing {k}; available: {list(proto)}")
targets = list(proto['targets'])
excluded = list(proto['excluded_columns'])
group = proto.get('group_column')
train = pd.read_parquet(p['train'])
val = pd.read_parquet(p['val'])

drop = set(targets) | set(excluded) | ({group} if group else set())
cand = [c for c in train.columns if c not in drop]

def is_keylike(s):
    nz = s.notna().sum()
    if nz == 0: return False
    if s.nunique(dropna=True) < nz: return False
    if pd.api.types.is_float_dtype(s): return False
    return True

keylike = [c for c in cand if is_keylike(train[c])]
cand = [c for c in cand if c not in keylike]

ph_tokens = ['?', '-', 'NA', 'N/A']
ph_count = {}
for c in cand:
    if pd.api.types.is_numeric_dtype(train[c]): continue
    s = train[c].astype('string').str.strip()
    m = s.isin(ph_tokens) | s.eq('')
    k = int(m.sum())
    if k: ph_count[c] = k
if ph_count:
    print(f"Placeholders in {len(ph_count)} cols: {sorted(ph_count)}")
    for c in ph_count:
        for df in (train, val):
            df[c] = df[c].astype('string').str.strip().replace(ph_tokens + [''], np.nan)

tcol = targets[0]
keep, excl_feat = [], []
for c in cand:
    a = train[c].astype('string').fillna('__NA__')
    b = train[tcol].astype('string').fillna('__NA__')
    if a.equals(b):
        excl_feat.append(c)
    else:
        keep.append(c)
for c in excl_feat:
    print(f"Excluded feature {c}: exact-equal-target")

n = len(train)
num_cols, cat_cols, drop_high, date_cols = [], [], [], []
for c in keep:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        num_cols.append(c); continue
    s2 = s.dropna().astype(str)
    if len(s2) > 0:
        try:
            pd.to_datetime(s2, errors='raise'); date_cols.append(c); continue
        except Exception:
            pass
    nu = int(s.nunique(dropna=True))
    if nu > 20 and nu > 0.05 * n:
        drop_high.append(c)
    else:
        cat_cols.append(c)

for c in date_cols:
    for df in (train, val):
        dt = pd.to_datetime(df[c], errors='coerce')
        df[c + '_hour'] = dt.dt.hour.astype(float)
        df[c + '_dow'] = dt.dt.dayofweek.astype(float)
        df[c + '_month'] = dt.dt.month.astype(float)
    num_cols.extend([c + '_hour', c + '_dow', c + '_month'])
if date_cols: print(f"Date cols derived: {len(date_cols)}")
if drop_high: print(f"Dropped high-cardinality: {sorted(drop_high)}")

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc', StandardScaler())])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
trs = []
if num_cols: trs.append(('num', num_pipe, num_cols))
if cat_cols: trs.append(('cat', cat_pipe, cat_cols))
pre = ColumnTransformer(trs, remainder='drop', verbose_feature_names_out=False)

n_out = len(num_cols) + sum(int(train[c].nunique(dropna=True)) + 1 for c in cat_cols)
if n_out <= 300:
    pre.set_output(transform='pandas')

feat_cols = num_cols + cat_cols
Xtr, Xva = train[feat_cols], val[feat_cols]
mlca_runtime.fit_record(pre, Xtr, 'train')
Xv = pre.transform(Xva)
shape_tr = Xtr.shape[0], n_out if n_out <= 300 else pre.transform(Xtr).shape[1]
print(f"Features: {len(num_cols)} num, {len(cat_cols)} cat; train {Xtr.shape} -> {shape_tr}; val -> ({Xva.shape[0]}, {shape_tr[1]})")

mlca_runtime.save_model(pre, 'preprocessor.pkl')
mlca_runtime.record_metric('n_features_in', len(feat_cols), 'train')
mlca_runtime.record_metric('n_encoded_features', n_out, 'train')
