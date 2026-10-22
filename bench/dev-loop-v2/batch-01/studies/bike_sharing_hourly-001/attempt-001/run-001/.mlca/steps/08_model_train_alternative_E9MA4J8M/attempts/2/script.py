import json
import pandas as pd
import numpy as np
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')
primary_metric = protocol['task']['primary_metric']

train = pd.read_parquet(inp['train']).copy()
val = pd.read_parquet(inp['val']).copy()

for df in (train, val):
    if 'dteday' in df.columns:
        dt = pd.to_datetime(df['dteday'], errors='coerce')
        df['dt_month'] = dt.dt.month.astype('float64')
        df['dt_dow'] = dt.dt.dayofweek.astype('float64')

num_cols, cat_cols = [], []
n = max(len(train), 1)
for c in train.columns:
    if c in targets or c in excluded or c == group or c == 'dteday':
        continue
    s = train[c]
    is_cat = (s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype)
              or str(s.dtype).startswith('string'))
    if is_cat:
        nu = s.nunique(dropna=True)
        if nu > 0.05 * n or nu > 50:
            continue
        cat_cols.append(c)
    else:
        if s.dtype == bool:
            s = s.astype('float64')
            train[c] = s
            val[c] = val[c].astype('float64')
        num_cols.append(c)

feature_cols = num_cols + cat_cols
X_tr = train[feature_cols]
X_va = val[feature_cols]
y_tr = train[targets]
y_va = val[targets]

transformers = []
if num_cols:
    transformers.append(('num', SimpleImputer(strategy='median'), num_cols))
if cat_cols:
    transformers.append(('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), cat_cols))

pre = ColumnTransformer(transformers, remainder='drop')
pre.set_output(transform='pandas')

pipe = Pipeline([
    ('pre', pre),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
])

fitted = mlca_runtime.fit_record(pipe, X_tr, partition='train', y=y_tr)

preds = fitted.predict(X_va)
if preds.ndim == 1:
    preds = preds.reshape(-1, 1)
rmses = []
for i, tname in enumerate(targets):
    rmse = float(np.sqrt(mean_squared_error(y_va.iloc[:, i], preds[:, i])))
    rmses.append(rmse)
    mlca_runtime.record_metric(f'{primary_metric}_{tname}', rmse, split='val')
mlca_runtime.record_metric(primary_metric, float(np.mean(rmses)), split='val')

mlca_runtime.save_model(fitted, 'alternative_model.pkl')
print(f"features={len(feature_cols)} num={len(num_cols)} cat={len(cat_cols)}")
print(f"train={X_tr.shape} val={X_va.shape} rmse={np.mean(rmses):.4f}")
