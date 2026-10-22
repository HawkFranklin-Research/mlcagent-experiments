import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')
primary_metric = protocol['task']['primary_metric']

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

with open(inp['05_preprocess_fit_preprocessing_5ED3RGCF/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inp['06_features_retain_features_YGN0TN24/selector_all.pkl'], 'rb') as f:
    sel = pickle.load(f)

if hasattr(pre, 'feature_names_in_'):
    feature_cols = [c for c in pre.feature_names_in_ if c in train.columns]
else:
    feature_cols = [c for c in train.columns
                    if c not in targets and c not in excluded and c != group]

X_tr = train[feature_cols].copy()
X_va = val[feature_cols].copy()
y_tr = train[targets].copy()
y_va = val[targets].copy()

steps = [('pre', pre)]
if sel is not None:
    steps.append(('sel', sel))
steps.append(('rf', RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1)))
pipe = Pipeline(steps)

fitted = mlca_runtime.fit_record(pipe, X_tr, partition='train', y=y_tr)

preds = fitted.predict(X_va)
if preds.ndim == 1:
    preds = preds.reshape(-1, 1)
for i, tname in enumerate(targets):
    rmse = float(np.sqrt(mean_squared_error(y_va.iloc[:, i], preds[:, i])))
    mlca_runtime.record_metric(f'{primary_metric}_{tname}', rmse, split='val')

mlca_runtime.save_model(fitted, 'alternative_model.pkl')
print(f"features={len(feature_cols)} train={X_tr.shape} val={X_va.shape}")
