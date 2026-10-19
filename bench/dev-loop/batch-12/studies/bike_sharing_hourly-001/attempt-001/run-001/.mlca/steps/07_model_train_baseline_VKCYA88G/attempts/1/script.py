import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
train = pd.read_parquet(paths['train'])
val = pd.read_parquet(paths['val'])
protocol = json.loads(paths['_protocol'].read_text())

targets = protocol['targets']
excluded = list(protocol['excluded_columns'])
group = protocol.get('group_column')

# RL-0007: one feature contract (targets, excluded, group, key-like cols removed)
feats = [c for c in train.columns if c not in targets and c not in excluded and c != group]
# RL-0010: drop raw datetime columns (already have yr/mnth/weekday/hr calendar parts)
feats = [c for c in feats if not pd.api.types.is_datetime64_any_dtype(train[c])]
if 'dteday' in feats:
    feats.remove('dteday')  # redundant sequential date key
# RL-0010: keep numeric only (no text/code columns present in this table)
feats = [c for c in feats if pd.api.types.is_numeric_dtype(train[c])]

pre = ColumnTransformer([('num', StandardScaler(), feats)], remainder='drop')
pre.set_output(transform='pandas')
reg = TransformedTargetRegressor(regressor=LinearRegression(),
                                 func=np.log1p, inverse_func=np.expm1)
pipe = Pipeline([('pre', pre), ('reg', reg)])
pipe.set_output(transform='pandas')

X_train = train[feats]
X_val = val[feats]
if len(targets) == 1:
    y_train = train[targets[0]]
    y_val_arr = val[targets[0]].to_numpy()
else:
    y_train = train[targets]
    y_val_arr = val[targets].to_numpy()

mlca_runtime.fit_record(pipe, X_train, 'train', y_train)

pred = pipe.predict(X_val)
if pred.ndim == 2 and pred.shape[1] == 1:
    pred = pred.ravel()
rmse_val = float(np.sqrt(np.mean((pred - y_val_arr) ** 2)))
mlca_runtime.record_metric('rmse', rmse_val, 'val')

pred_tr = pipe.predict(X_train)
if pred_tr.ndim == 2 and pred_tr.shape[1] == 1:
    pred_tr = pred_tr.ravel()
y_train_arr = y_train.to_numpy()
if y_train_arr.ndim == 2 and y_train_arr.shape[1] == 1:
    y_train_arr = y_train_arr.ravel()
rmse_tr = float(np.sqrt(np.mean((pred_tr - y_train_arr) ** 2)))
mlca_runtime.record_metric('rmse', rmse_tr, 'train')

mlca_runtime.save_model(pipe, 'model.pkl')

print('n_train', len(X_train), 'n_val', len(X_val), 'n_features', len(feats))
print('val_rmse', round(rmse_val, 4))
