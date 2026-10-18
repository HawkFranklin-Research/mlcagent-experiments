import mlca_runtime, json
import pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
prot = json.loads(paths['_protocol'].read_text())
train = pd.read_parquet(paths['train'])
targets, excluded = prot['targets'], prot['excluded_columns']
group_col = prot.get('group_column')
drop_all = set(targets) | set(excluded) | ({group_col} if group_col else set())
candidates = [c for c in train.columns if c not in drop_all]
n = len(train)
y = train[targets[0]]
y_vals = y.astype('float64').values

ph = {'?', '-', 'NA', 'N/A', ''}
ph_hits = {}
for c in candidates:
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        st = s.astype('string').str.strip()
        toks = set(st.dropna().unique()) & ph
        if toks:
            ph_hits[c] = toks
            train[c] = st.where(~st.isin(list(ph)), other=pd.NA).astype(object)
if ph_hits:
    print(f"Placeholders {sorted(set().union(*ph_hits.values()))} in {len(ph_hits)} cols")

numeric_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(train[c])]
derived = set()
for c in numeric_cols:
    if np.array_equal(train[c].astype('float64').fillna(0).values, y_vals):
        derived.add(c)
nonneg = [c for c in numeric_cols if (train[c].fillna(0) >= 0).all() and c not in derived]
for i, ci in enumerate(nonneg):
    ai = train[ci].astype('float64').fillna(0).values
    for j in range(i+1, len(nonneg)):
        cj = nonneg[j]
        s2 = ai + train[cj].astype('float64').fillna(0).values
        if np.allclose(s2, y_vals, rtol=0, atol=1e-8):
            derived.update([ci, cj]); continue
        for k in range(j+1, len(nonneg)):
            s3 = s2 + train[nonneg[k]].astype('float64').fillna(0).values
            if np.allclose(s3, y_vals, rtol=0, atol=1e-8):
                derived.update([ci, cj, nonneg[k]])

key_like = set()
for c in candidates:
    s = train[c]
    if s.dtype.kind == 'f': continue
    if pd.api.types.is_numeric_dtype(s) or s.dtype == object:
        nn = s.dropna()
        if len(nn) == n and nn.nunique() == n: key_like.add(c)

excluded_feats = derived | key_like
if excluded_feats: print(f"Excluded features: {sorted(excluded_feats)}")

if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
    warn = []
    for c in candidates:
        if c in excluded_feats or not pd.api.types.is_numeric_dtype(train[c]): continue
        xv = train[c].astype('float64').fillna(train[c].median()).values
        if np.std(xv) == 0: continue
        if abs(np.corrcoef(xv, y_vals)[0, 1]) >= 0.98: warn.append(c)
    if warn: print(f"High corr kept: {warn}")

num_cols, cat_cols = [], []
for c in candidates:
    if c in excluded_feats: continue
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        try:
            if pd.to_datetime(s, errors='coerce').notna().mean() > 0.9: continue
        except Exception: pass
        if s.dropna().nunique() > 0.05 * n: continue
        cat_cols.append(c)
    elif pd.api.types.is_integer_dtype(s):
        nu = s.dropna().nunique()
        if nu == 2 and set(s.dropna().unique()).issubset({0, 1}): num_cols.append(c)
        elif nu <= 12: cat_cols.append(c)
        else: num_cols.append(c)
    else:
        num_cols.append(c)

width = len(num_cols) + sum(train[c].dropna().nunique() for c in cat_cols)
if width <= 300:
    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]).set_output(transform='pandas')
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]).set_output(transform='pandas')
    pre = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)], remainder='drop')
    pre.set_output(transform='pandas')
else:
    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore'))])
    pre = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)], remainder='drop')

X_train = train[num_cols + cat_cols]
mlca_runtime.fit_record(pre, X_train, 'train')
mlca_runtime.save_model(pre, 'preprocessor.pkl')
print(f"Fit preprocessor on {n} rows: {len(num_cols)} numeric, {len(cat_cols)} categorical")
