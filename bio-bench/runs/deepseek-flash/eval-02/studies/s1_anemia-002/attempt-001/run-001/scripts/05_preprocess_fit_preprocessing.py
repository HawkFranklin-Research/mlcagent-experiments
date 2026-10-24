import mlca_runtime
import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
protocol = json.loads(inputs['_protocol'].read_text())

targets = list(protocol['targets'])
excluded = list(protocol.get('excluded_columns', []))
group = protocol.get('group_column')
target_col = targets[0] if len(targets) == 1 else None

placeholders = {'?', '-', 'NA', 'N/A'}
def clean_val(v):
    if isinstance(v, str):
        s = v.strip()
        if s in placeholders or s == '':
            return np.nan
        return s
    return v

ph_cols, ph_tokens = [], set()
for c in train.columns:
    if train[c].dtype == object or isinstance(train[c].dtype, pd.CategoricalDtype):
        s = train[c].dropna().astype(str)
        if len(s) == 0:
            continue
        stripped = s.str.strip()
        toks = set(stripped[stripped.isin(placeholders) | (stripped == '')].unique())
        if toks or (s != stripped).any():
            ph_cols.append(c)
            ph_tokens.update(toks)
if ph_cols:
    print(f"placeholders {sorted(t for t in ph_tokens if t)} in {len(ph_cols)} columns")
    for c in ph_cols:
        train[c] = train[c].map(clean_val)
        if c in val.columns:
            val[c] = val[c].map(clean_val)

def is_key_like(s):
    if s.nunique(dropna=True) != len(s):
        return False
    if s.dtype == object:
        return True
    if pd.api.types.is_integer_dtype(s):
        v = np.sort(s.dropna().astype(float).values)
        return len(v) > 1 and (v[-1] - v[0] + 1) == len(v)
    return False

all_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
feature_cols = [c for c in all_cols if not is_key_like(train[c])]
if len(feature_cols) < len(all_cols):
    print(f"dropped {len(all_cols) - len(feature_cols)} key-like columns")

for c in list(feature_cols):
    if pd.api.types.is_datetime64_any_dtype(train[c]):
        tr = train[c]
        for part, fn_tr, fn_v in (('year', lambda x: x.dt.year, lambda x: x.dt.year),
                                  ('month', lambda x: x.dt.month, lambda x: x.dt.month),
                                  ('dow', lambda x: x.dt.dayofweek, lambda x: x.dt.dayofweek)):
            train[f"{c}_{part}"] = fn_tr(tr)
            if c in val.columns:
                val[f"{c}_{part}"] = fn_v(pd.to_datetime(val[c], errors='coerce'))
        feature_cols.remove(c)
        feature_cols += [f"{c}_year", f"{c}_month", f"{c}_dow"]

if target_col is not None:
    y_raw = train[target_col]
    y_bin = None
    if pd.api.types.is_numeric_dtype(y_raw) and y_raw.dropna().isin([0, 1]).all():
        y_bin = y_raw.astype(float)
    elif y_raw.dtype == object or isinstance(y_raw.dtype, pd.CategoricalDtype):
        u = y_raw.dropna().unique()
        if len(u) == 2:
            y_bin = y_raw.map({u[0]: 0.0, u[1]: 1.0})
    if y_bin is not None:
        to_drop = set()
        for c in feature_cols:
            if pd.api.types.is_numeric_dtype(train[c]) and train[c].astype(float).equals(y_bin):
                to_drop.add(c)
                print(f"drop {c}: exact match with target")
        nonneg = [c for c in feature_cols if c not in to_drop
                  and pd.api.types.is_numeric_dtype(train[c])
                  and (train[c].dropna() >= 0).all()]
        for k in (1, 2, 3):
            for combo in combinations(nonneg, k):
                if set(combo) & to_drop:
                    continue
                if train[list(combo)].astype(float).sum(axis=1).equals(y_bin):
                    to_drop.update(combo)
                    print(f"drop {'+'.join(combo)}: row sum equals target")
        if to_drop:
            feature_cols = [c for c in feature_cols if c not in to_drop]
        warns = []
        for c in feature_cols:
            if pd.api.types.is_numeric_dtype(train[c]):
                pair = pd.concat([train[c].astype(float), y_bin], axis=1).dropna()
                if len(pair) > 2:
                    r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
                    if abs(r) >= 0.98:
                        warns.append(c)
        if warns:
            print(f"high |corr|>=0.98 with target (kept): {warns}")

max_cat_card = max(2, int(0.05 * len(train)))
num_cols, cat_cols, drop_cat = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    elif s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype):
        (cat_cols if s.nunique(dropna=True) <= max_cat_card else drop_cat).append(c)
    else:
        try:
            pd.to_numeric(s, errors='raise')
            num_cols.append(c)
        except Exception:
            drop_cat.append(c)
if drop_cat:
    print(f"dropping high-cardinality text: {len(drop_cat)} cols")
    feature_cols = [c for c in feature_cols if c not in drop_cat]

num_pipe = Pipeline([('impute', SimpleImputer(strategy='median')),
                     ('scale', StandardScaler())])
cat_pipe = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
transformers = []
if num_cols:
    transformers.append(('num', num_pipe, num_cols))
if cat_cols:
    transformers.append(('cat', cat_pipe, cat_cols))
pre = ColumnTransformer(transformers, remainder='drop')

n_out = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
if n_out <= 300:
    pre.set_output(transform='pandas')

X_train = train[feature_cols]
pre_fitted = mlca_runtime.fit_record(pre, X_train, 'train')
X_val = val[[c for c in feature_cols if c in val.columns]]
Xt = pre_fitted.transform(X_val)
print(f"features={len(feature_cols)}, encoded={Xt.shape[1]}")

mlca_runtime.save_model(pre_fitted, 'preprocessor.pkl')
mlca_runtime.record_metric("n_features_in", len(feature_cols), "train")
mlca_runtime.record_metric("n_features_out", Xt.shape[1], "train")
