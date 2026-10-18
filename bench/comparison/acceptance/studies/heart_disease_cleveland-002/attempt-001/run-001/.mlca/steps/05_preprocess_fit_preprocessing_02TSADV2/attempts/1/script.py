import json
from itertools import combinations
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
for k in ('targets', 'excluded_columns', 'task'):
    if k not in protocol:
        raise KeyError(f"protocol missing '{k}'; available keys: {list(protocol.keys())}")
targets = list(protocol['targets'])
excluded = list(protocol['excluded_columns'])
group_col = protocol.get('group_column')
target = targets[0]

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

PH = {'?', '-', 'NA', 'N/A', ''}
def clean_text(df):
    df = df.copy(); hits, ncols = set(), 0
    for c in df.columns:
        s = df[c]
        if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            continue
        stripped = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        ph = s.map(lambda v: isinstance(v, str)) & stripped.map(lambda v: isinstance(v, str) and v in PH)
        if ph.any():
            ncols += 1
            for u in s[ph].unique():
                hits.add(str(u).strip())
            stripped = stripped.mask(ph, np.nan)
        df[c] = stripped
    return df, hits, ncols

train, ph_hits, ph_cols = clean_text(train)
val, _, _ = clean_text(val)

n = len(train)
candidates = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
feats = []
for c in candidates:
    s = train[c]; nun = s.nunique(dropna=True)
    if (s.dtype == object or pd.api.types.is_string_dtype(s)) and nun == n:
        continue
    if pd.api.types.is_integer_dtype(s) and nun == n:
        continue
    feats.append(c)

y = train[target]
derived = []
for c in list(feats):
    if train[c].equals(y):
        derived.append((c, 'equals target')); feats.remove(c)

if pd.api.types.is_numeric_dtype(y):
    nn = [c for c in feats if pd.api.types.is_numeric_dtype(train[c]) and train[c].notna().all() and (train[c] >= 0).all()]
    yv = y.to_numpy()
    for size in (2, 3):
        for combo in combinations(nn, size):
            if np.array_equal(sum(train[c] for c in combo).to_numpy(), yv):
                for c in combo:
                    if c in feats:
                        feats.remove(c); derived.append((c, f'row-sum of {combo}'))

warn = []
if pd.api.types.is_numeric_dtype(y):
    yf = y.to_numpy(dtype=float)
    for c in feats:
        if pd.api.types.is_numeric_dtype(train[c]):
            r = np.corrcoef(train[c].to_numpy(dtype=float), yf)[0, 1]
            if abs(r) >= 0.98:
                warn.append(c)

num_feats, cat_feats, drop_feats, date_feats = [], [], [], []
for c in feats:
    s = train[c]
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        num_feats.append(c); continue
    parsed = None
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        try:
            parsed = pd.to_datetime(s, errors='coerce')
        except Exception:
            parsed = None
    if parsed is not None and len(parsed) > 0 and parsed.notna().all():
        date_feats.append(c); continue
    if s.nunique(dropna=True) > 0.05 * n:
        drop_feats.append(c)
    else:
        cat_feats.append(c)
        train[c] = train[c].astype(object); val[c] = val[c].astype(object)

for c in date_feats:
    for df in (train, val):
        p = pd.to_datetime(df[c], errors='coerce')
        df[f"{c}_dow"] = p.dt.dayofweek.astype(float)
        df[f"{c}_month"] = p.dt.month.astype(float)
        df[f"{c}_hour"] = p.dt.hour.fillna(0).astype(float)
    num_feats += [f"{c}_dow", f"{c}_month", f"{c}_hour"]

final_feats = num_feats + cat_feats
width = len(num_feats) + sum(int(train[c].nunique(dropna=True)) for c in cat_feats)

pre = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_feats),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_feats),
    ],
    remainder='drop', verbose_feature_names_out=False,
)
if width <= 300:
    pre.set_output(transform='pandas')

fitted = mlca_runtime.fit_record(pre, train[final_feats], 'train')
_Xv = fitted.transform(val[final_feats])

mlca_runtime.save_model(fitted, 'preprocessor.pkl')
mlca_runtime.record_metric('n_features_selected', len(final_feats), 'train')
mlca_runtime.record_metric('n_derived_excluded', len(derived), 'train')

print(f"train {train.shape} val {val.shape}; placeholders {sorted(ph_hits)} in {ph_cols} cols")
print(f"derived excluded: {derived}")
print(f"corr>=0.98 warning: {warn}")
print(f"num={len(num_feats)} cat={len(cat_feats)} dropped={drop_feats} date={date_feats}")
print(f"encoded width={_Xv.shape[1]} val_rows={_Xv.shape[0]}")
