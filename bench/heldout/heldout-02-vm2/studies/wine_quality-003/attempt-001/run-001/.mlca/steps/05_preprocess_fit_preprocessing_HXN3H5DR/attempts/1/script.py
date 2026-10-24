import mlca_runtime as mlca
import json, itertools
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca.set_seed(42)
inp = mlca.inputs()
train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
protocol = json.loads(inp['_protocol'].read_text())

try:
    targets = protocol['targets']
    excluded = protocol['excluded_columns']
    group_column = protocol.get('group_column')
    primary_metric = protocol['task']['primary_metric']
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; top-level keys: {list(protocol.keys())}")

reserved = set(targets) | set(excluded)
if group_column:
    reserved.add(group_column)
feat = [c for c in train.columns if c not in reserved]

# drop object columns whose values are all distinct (row ids)
feat = [c for c in feat if not (train[c].dtype == object and train[c].notna().all()
                                and train[c].nunique() == len(train))]

# datetime handling: derive numeric parts, then drop raw column
def parse_dt(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if s.dtype == object or str(s.dtype) in ('string', 'str'):
        try:
            p = pd.to_datetime(s, errors='raise')
            if p.notna().mean() > 0.5:
                return p
        except Exception:
            return None
    return None

new_dt = []
for c in list(feat):
    tr_p = parse_dt(train[c])
    if tr_p is None:
        continue
    va_p = parse_dt(val[c])
    for name, fn in [('dow', lambda x: x.dt.dayofweek),
                     ('month', lambda x: x.dt.month),
                     ('hour', lambda x: x.dt.hour),
                     ('days', lambda x: (x - pd.Timestamp('1970-01-01')).dt.days)]:
        nc = f'{c}_{name}'
        train[nc] = fn(tr_p); val[nc] = fn(va_p)
        new_dt.append(nc)
    feat.remove(c)
feat = feat + new_dt

# placeholder-token cleaning for text columns
placeholders = {'?', '-', 'NA', 'N/A'}
hit_cols = []
for c in list(feat):
    s = train[c]
    if not (s.dtype == object or str(s.dtype) in ('string', 'str') or isinstance(s.dtype, pd.CategoricalDtype)):
        continue
    def clean(series):
        o = series.astype(object)
        stripped = o.map(lambda v: v.strip() if isinstance(v, str) else v)
        m = stripped.map(lambda v: isinstance(v, str) and (v.strip() == '' or v.strip() in placeholders)).astype(bool)
        return stripped.where(~m, np.nan), bool(m.any())
    train[c], h = clean(train[c])
    val[c], _ = clean(val[c])
    if h:
        hit_cols.append(c)
print(f"Placeholders in {len(hit_cols)} text columns: {hit_cols}" if hit_cols else "No placeholder tokens found")

# screen for confirmed derivations of the target
y_tr = train[targets[0]] if len(targets) == 1 else None
derived, reasons = [], {}
if y_tr is not None and pd.api.types.is_numeric_dtype(y_tr):
    num_feats = [c for c in feat if pd.api.types.is_numeric_dtype(train[c]) and train[c].notna().all()]
    for c in num_feats:
        if (train[c] == y_tr).all():
            derived.append(c); reasons[c] = 'exact equality with target'
    cand = [c for c in num_feats if (train[c] >= 0).all() and c not in derived]
    for k in (2, 3):
        for combo in itertools.combinations(cand, k):
            tot = sum(train[c] for c in combo)
            if (tot == y_tr).all():
                for c in combo:
                    if c not in derived:
                        derived.append(c); reasons[c] = 'row sum = target'
if derived:
    print(f"Excluded {len(derived)} derived: " + ", ".join(f'{c} ({reasons[c]})' for c in derived))
    feat = [c for c in feat if c not in derived]

if y_tr is not None and pd.api.types.is_numeric_dtype(y_tr):
    yv = y_tr.astype(float).values
    warn = []
    for c in feat:
        x = train[c]
        if not pd.api.types.is_numeric_dtype(x):
            continue
        xv = x.astype(float).values
        if np.nanstd(xv) == 0:
            continue
        if abs(np.corrcoef(xv, yv)[0, 1]) >= 0.98:
            warn.append(c)
    if warn:
        print(f"Correlation >=0.98 with target (kept): {warn}")

# numeric vs categorical split
thr = 0.05 * len(train)
numeric_cols, cat_cols, high_card = [], [], []
for c in feat:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    elif s.nunique(dropna=True) > thr:
        high_card.append(c)
    else:
        cat_cols.append(c)
if high_card:
    print(f"Dropped {len(high_card)} high-cardinality text columns: {high_card}")

for c in cat_cols:
    train[c] = train[c].astype(object).where(train[c].notna(), np.nan)
    val[c] = val[c].astype(object).where(val[c].notna(), np.nan)

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc', StandardScaler())])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
num_pipe.set_output(transform='pandas')
cat_pipe.set_output(transform='pandas')

pre = ColumnTransformer([('num', num_pipe, numeric_cols),
                         ('cat', cat_pipe, cat_cols)],
                        remainder='drop', verbose_feature_names_out=False)
pre.set_output(transform='pandas')

X_train = train[feat].copy()
X_val = val[feat].copy()
fitted = mlca.fit_record(pre, X_train, 'train')
Xt = fitted.transform(X_val)
print(f"Feature frame {X_train.shape} -> encoded {Xt.shape[1]} cols; numeric {len(numeric_cols)}, cat {len(cat_cols)}")

mlca.save_model(fitted, 'preprocessor.pkl')
