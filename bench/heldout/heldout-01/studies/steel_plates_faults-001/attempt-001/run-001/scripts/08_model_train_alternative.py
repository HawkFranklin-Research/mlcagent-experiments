import mlca_runtime, json
import pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = list(protocol['targets'])
excluded = list(protocol['excluded_columns'])
group = protocol.get('group_column')
primary = protocol['task']['primary_metric']

train = pd.read_parquet(inputs['train'])
y = train[targets].copy()
n = len(train)
drop = set(targets) | set(excluded)
if group: drop.add(group)

feat_cols = []
for c in train.columns:
    if c in drop: continue
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=True) == s.notna().sum() and s.nunique(dropna=True) > max(20, 0.05*n):
            continue
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=True) == n and n > 20:
            continue
    feat_cols.append(c)

X = train[feat_cols].copy()

numeric_cols, cat_cols, date_cols = [], [], []
for c in feat_cols:
    s = X[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    else:
        if s.nunique(dropna=True) <= max(20, 0.05*n):
            cat_cols.append(c)

for c in date_cols:
    d = pd.to_datetime(X[c], errors='coerce')
    for part, getter in [('dayofweek', d.dt.dayofweek), ('month', d.dt.month), ('day', d.dt.day)]:
        X[f'{c}_{part}'] = getter.astype(float)
        numeric_cols.append(f'{c}_{part}')
    X = X.drop(columns=[c])

n_enc = len(numeric_cols) + sum(X[c].nunique(dropna=True) for c in cat_cols)
use_pandas = n_enc <= 300

transformers = []
if numeric_cols:
    transformers.append(('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                                          ('sc', StandardScaler())]), numeric_cols))
if cat_cols:
    transformers.append(('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                                          ('ohe', OneHotEncoder(handle_unknown='ignore',
                                                                sparse_output=not use_pandas))]), cat_cols))
pre = ColumnTransformer(transformers, remainder='drop')
if use_pandas:
    pre.set_output(transform='pandas')

model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
pipe = Pipeline([('pre', pre), ('model', model)])

mlca_runtime.fit_record(pipe, X, partition='train', y=y)

pred = pipe.predict(X)
f1s = [f1_score(y[t].values, pred[:, i], average='macro', zero_division=0) for i, t in enumerate(targets)]
mlca_runtime.record_metric('train_macro_f1', float(np.mean(f1s)), 'train')
for t, v in zip(targets, f1s):
    mlca_runtime.record_metric(f'train_f1_{t}', float(v), 'train')

mlca_runtime.save_model(pipe, 'alternative_model.pkl')
print('features:', len(X.columns), 'encoded_cols:', n_enc)
print('train macro_f1:', round(float(np.mean(f1s)), 4))
