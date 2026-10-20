import mlca_runtime, json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths['_protocol'].read_text())
targets = proto['targets']
excluded = list(proto.get('excluded_columns', []))
group_col = proto.get('group_column', None)

train = pd.read_parquet(paths['train'])
val = pd.read_parquet(paths['val'])

feat_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]

cat_cols, num_cols = [], []
n = len(train)
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    elif pd.api.types.is_datetime64_any_dtype(s):
        train[c] = s.astype('int64')
        val[c] = val[c].astype('int64')
        num_cols.append(c)
    else:
        if s.nunique(dropna=True) > max(10, 0.05 * n):
            continue
        cat_cols.append(c)
feat_cols = num_cols + cat_cols

pre = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), cat_cols),
])

pipe = Pipeline([
    ('pre', pre),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42)),
])

mlca_runtime.fit_record(pipe, train[feat_cols], 'train', train[targets])
mlca_runtime.save_model(pipe, 'alternative_model.pkl')

pred = np.asarray(pipe.predict(val[feat_cols]))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
y_val = val[targets]
rmses = []
for i, t in enumerate(targets):
    yt = y_val[t].to_numpy()
    m = float(np.sqrt(np.mean((pred[:, i] - yt) ** 2)))
    rmses.append(m)
    mlca_runtime.record_metric(f'rmse_{t}', m, 'val')
mlca_runtime.record_metric('rmse_mean', float(np.mean(rmses)), 'val')
print(f'features: {len(feat_cols)} num={len(num_cols)} cat={len(cat_cols)}')
print(f'val rmse mean: {np.mean(rmses):.4f}')
