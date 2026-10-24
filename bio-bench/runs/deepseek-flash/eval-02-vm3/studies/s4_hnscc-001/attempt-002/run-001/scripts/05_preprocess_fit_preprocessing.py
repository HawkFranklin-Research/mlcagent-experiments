import mlca_runtime, json, re, warnings
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
warnings.filterwarnings('ignore')
mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs['train']); val = pd.read_parquet(inputs['val'])
protocol = json.loads(inputs['_protocol'].read_text())
targets = list(protocol['targets']); excluded = list(protocol['excluded_columns'])
group_col = protocol.get('group_column', None)

# Placeholder cleanup: strip whitespace, convert placeholder tokens to NaN
ph_cols = []
for c in train.columns:
    if train[c].dtype == object or train[c].dtype.name in ('string', 'object'):
        train[c] = train[c].astype(object).apply(lambda v: v.strip() if isinstance(v, str) else v)
        val[c] = val[c].astype(object).apply(lambda v: v.strip() if isinstance(v, str) else v)
        for tok in ['?', '-', 'NA', 'N/A', '']:
            tr_m = train[c].apply(lambda v: isinstance(v, str) and v == tok)
            if tr_m.any():
                train.loc[tr_m, c] = np.nan
                val.loc[val[c].apply(lambda v: isinstance(v, str) and v == tok), c] = np.nan
        if train[c].isna().sum() > 0:
            ph_cols.append(c)
if ph_cols:
    print(f"Placeholders in {len(ph_cols)} cols: {ph_cols[:5]}")

# Candidate feature columns
feat = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
keep = []
for c in feat:
    s = train[c]
    if s.dtype == object or s.dtype.name == 'string':
        if s.notna().sum() > 0 and s.nunique(dropna=True) == s.notna().sum():
            continue
    if pd.api.types.is_integer_dtype(s) and re.search(r'id|code|key|sample', c, re.I):
        if s.notna().sum() > 0 and s.nunique(dropna=True) == s.notna().sum():
            continue
    keep.append(c)
feat = keep

# Derivation screen: exact equality and row sums of up to three non-negative columns
num_f = [c for c in feat if pd.api.types.is_numeric_dtype(train[c])]
Xn = train[num_f].to_numpy(dtype=float)
excl_map = {}
for tgt in targets:
    if not pd.api.types.is_numeric_dtype(train[tgt]):
        continue
    yv = train[tgt].to_numpy(dtype=float)
    eq = np.all(np.isclose(Xn, yv[:, None], equal_nan=True, rtol=1e-9, atol=1e-9), axis=0)
    for i, c in enumerate(num_f):
        if eq[i]:
            excl_map.setdefault(c, f'exact=={tgt}')
    tmin = np.nanmin(yv)
    cand = [i for i in range(len(num_f))
            if np.nanmin(Xn[:, i]) >= 0 and np.nanmax(Xn[:, i]) >= tmin / 2]
    if 2 <= len(cand) <= 300:
        for a in range(len(cand)):
            for b in range(a + 1, len(cand)):
                s2 = Xn[:, cand[a]] + Xn[:, cand[b]]
                if np.all(np.isclose(s2, yv, equal_nan=True, rtol=1e-9, atol=1e-9)):
                    excl_map.setdefault(num_f[cand[a]], f'pair-sum=={tgt}')
                    excl_map.setdefault(num_f[cand[b]], f'pair-sum=={tgt}')
    if 3 <= len(cand) <= 40:
        for a in range(len(cand)):
            for b in range(a + 1, len(cand)):
                for d in range(b + 1, len(cand)):
                    s3 = Xn[:, cand[a]] + Xn[:, cand[b]] + Xn[:, cand[d]]
                    if np.all(np.isclose(s3, yv, equal_nan=True, rtol=1e-9, atol=1e-9)):
                        for ix in (a, b, d):
                            excl_map.setdefault(num_f[cand[ix]], f'triple-sum=={tgt}')
for c, r in list(excl_map.items())[:10]:
    print(f"Excluded feature {c}: {r}")
feat = [c for c in feat if c not in excl_map]

# Correlation warning against numeric (non-binary) targets
wcol = []
for tgt in targets:
    if not pd.api.types.is_numeric_dtype(train[tgt]):
        continue
    if train[tgt].nunique() <= 2:
        continue
    y = train[tgt].astype(float)
    for c in num_f:
        if c not in feat:
            continue
        try:
            r = np.corrcoef(train[c].astype(float), y)[0, 1]
        except Exception:
            continue
        if not np.isnan(r) and abs(r) >= 0.98:
            wcol.append(c)
if wcol:
    print(f"Corr>=0.98 with numeric target (kept, warning): {list(dict.fromkeys(wcol))[:10]}")

# Sort columns by type
num_c, cat_c, dt_c, drop_c = [], [], [], []
for c in feat:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dt_c.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        num_c.append(c)
    elif s.dtype == object or s.dtype.name == 'string':
        if s.nunique(dropna=True) > 0.05 * len(train):
            drop_c.append(c)
        else:
            cat_c.append(c)
    else:
        num_c.append(c)
for c in drop_c:
    print(f"Drop high-cardinality column {c}")
for c in dt_c:
    print(f"DateTime column {c} -> derive days elapsed")

# Datetime derived numeric features
for c in list(dt_c):
    train[c] = pd.to_datetime(train[c], errors='coerce')
    val[c] = pd.to_datetime(val[c], errors='coerce')
    ref = train[c].min()
    dcol = c + '_days_elapsed'
    train[dcol] = (train[c] - ref).dt.days.astype(float)
    val[dcol] = (val[c] - ref).dt.days.astype(float)
    num_c.append(dcol)

feat_final = [c for c in feat if c not in drop_c and c not in dt_c] + [c + '_days_elapsed' for c in dt_c]

# Build preprocessing ColumnTransformer
pre = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median', fill_value=0)),
                      ('sc', StandardScaler())]), num_c),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_c),
], remainder='drop')

width = len(num_c) + sum(train[c].nunique(dropna=True) for c in cat_c)
if width <= 300:
    pre.set_output(transform='pandas')
    print(f"Pandas output (est. width={width})")
else:
    print(f"Sparse output (est. width={width})")

Xtr = train[feat_final]
Xva = val[feat_final]
mlca_runtime.fit_record(pre, Xtr, 'train')
Xt = pre.transform(Xva)
print(f"Fit train rows={len(Xtr)}, val rows={len(Xva)}, transformed shape={getattr(Xt, 'shape', 'NA')}")
mlca_runtime.save_model(pre, 'preprocessor.pkl')
