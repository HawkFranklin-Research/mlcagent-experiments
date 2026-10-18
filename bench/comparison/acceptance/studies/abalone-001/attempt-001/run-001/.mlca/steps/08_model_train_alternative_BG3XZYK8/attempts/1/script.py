import mlca_runtime, json, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs['_protocol'].read_text())
for k in ('targets', 'excluded_columns', 'group_column', 'task'):
    if k not in proto:
        raise KeyError(f"missing key {k}; available: {list(proto.keys())}")
targets = list(proto['targets'])
excluded = list(proto['excluded_columns'])
group = proto['group_column']
task = proto['task']
metric_name = task['primary_metric']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

reserved = set(targets) | set(excluded)
if group:
    reserved.add(group)

feature_cols = []
for c in train.columns:
    if c in reserved:
        continue
    s = train[c]
    if s.nunique(dropna=False) == len(s) and (s.dtype == object
            or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)):
        continue
    feature_cols.append(c)

X_tr = train[feature_cols]
y_tr = train[targets[0]] if len(targets) == 1 else train[targets]
X_va = val[feature_cols]

cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(X_tr[c])]
num_cols = [c for c in feature_cols if c not in cat_cols]

pre = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ('num', 'passthrough', num_cols),
], remainder='drop')
pre.set_output(transform='pandas')
pipe = Pipeline([
    ('pre', pre),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
])

fitted = mlca_runtime.fit_record(pipe, X_tr, 'train', y=y_tr)
mlca_runtime.save_model(fitted, 'alternative_model.pkl')

def rmse(yt, yp):
    return float(np.sqrt(mean_squared_error(yt, yp)))

pred_tr = fitted.predict(X_tr)
pred_va = fitted.predict(X_va)

if len(targets) == 1:
    mlca_runtime.record_metric(metric_name, rmse(y_tr, pred_tr), 'train')
    mlca_runtime.record_metric(metric_name, rmse(val[targets[0]], pred_va), 'val')
else:
    per_t = [rmse(val[t], pred_va[:, i]) for i, t in enumerate(targets)]
    for t, v in zip(targets, per_t):
        mlca_runtime.record_metric(f"{metric_name}_{t}", v, 'val')
    mlca_runtime.record_metric(metric_name, float(np.mean(per_t)), 'val')

print(f"n_features={len(feature_cols)} cat={len(cat_cols)} num={len(num_cols)}")
print(f"train_rows={len(X_tr)} val_rows={len(X_va)}")
