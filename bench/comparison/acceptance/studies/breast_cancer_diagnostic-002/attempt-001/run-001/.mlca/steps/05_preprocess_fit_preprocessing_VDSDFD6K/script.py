import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol['targets']
excluded = protocol['excluded_columns'] or []
group_col = protocol.get('group_column')
target = targets[0]

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)

feat_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if s.dtype == object or str(s.dtype).startswith('str'):
        if s.nunique(dropna=True) == len(s):
            continue
    elif pd.api.types.is_integer_dtype(s) and len(s) > 1:
        if s.nunique(dropna=True) == len(s):
            continue
    feat_cols.append(c)

y = train[target]
deriv = []
for c in list(feat_cols):
    s = train[c]
    if s.dtype == object and y.dtype == object:
        same = (s.fillna('__NA__') == y.fillna('__NA__')).all()
        if same:
            deriv.append(c)
    elif pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(y):
        m = s.notna() & y.notna()
        if m.all() and (s == y).all():
            deriv.append(c)
for c in deriv:
    print(f"excluded derivation: {c} (exact equality with target)")
feat_cols = [c for c in feat_cols if c not in deriv]

ph_tokens = {'?', '-', 'NA', 'N/A', ''}
ph_cols = 0
for c in feat_cols:
    s = train[c]
    if s.dtype == object:
        if s.astype(str).str.strip().isin(ph_tokens).any():
            ph_cols += 1
print(f"text placeholders found in {ph_cols} column(s)")

n = len(train)
numeric_cols, categorical_cols = [], []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    else:
        nun = s.nunique(dropna=True)
        if nun <= max(2, int(0.05 * n)):
            categorical_cols.append(c)
        else:
            print(f"drop high-cardinality column: {c}")

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc', StandardScaler())])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer([('num', num_pipe, numeric_cols),
                                   ('cat', cat_pipe, categorical_cols)])

n_cat_out = sum(train[c].nunique(dropna=True) + 1 for c in categorical_cols)
if len(numeric_cols) + n_cat_out <= 300:
    try:
        preprocessor.set_output(transform='pandas')
    except Exception:
        pass

X_train = train[feat_cols]
fitted = mlca_runtime.fit_record(preprocessor, X_train, 'train')
Xt_val = fitted.transform(val[feat_cols])
print(f"num={len(numeric_cols)} cat={len(categorical_cols)} val_out={getattr(Xt_val, 'shape', None)}")

mlca_runtime.save_model(fitted, 'preprocessor.pkl')
