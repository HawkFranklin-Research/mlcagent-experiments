import mlca_runtime
import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

inputs = mlca_runtime.inputs()
mlca_runtime.set_seed(42)

protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column', None)
task = protocol['task']
metric_name = task['primary_metric']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
num_cols, cat_cols = [], []
for c in cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    elif s.nunique(dropna=False) <= max(5, int(0.05 * len(train))):
        cat_cols.append(c)

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
pre = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
], remainder='drop')
try:
    pre.set_output(transform='pandas')
except Exception:
    pass

model = Pipeline([('pre', pre), ('reg', LinearRegression())])

X_train = train[cols]
Y_train = train[targets].astype(float)
model = mlca_runtime.fit_record(model, X_train, 'train', y=Y_train)

X_val = val[cols]
Y_val = val[targets].astype(float)
pred = model.predict(X_val)

rmses = []
for i, t in enumerate(targets):
    rmse = float(np.sqrt(mean_squared_error(Y_val[t], pred[:, i])))
    rmses.append(rmse)
    mlca_runtime.record_metric(f'{t}_rmse', rmse, 'val')
mean_rmse = float(np.mean(rmses))
mlca_runtime.record_metric('mean_rmse', mean_rmse, 'val')

mlca_runtime.save_model(model, 'model.pkl')
print(f"train rows={len(train)} val rows={len(val)} features={len(cols)}")
print(f"val mean RMSE={mean_rmse:.4f}")
