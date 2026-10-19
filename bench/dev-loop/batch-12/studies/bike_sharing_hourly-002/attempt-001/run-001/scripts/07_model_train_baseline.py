import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())

for key in ['targets', 'excluded_columns', 'task']:
    if key not in protocol:
        raise KeyError(f"Missing _protocol[{key!r}]; available: {list(protocol)}")
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')
task = protocol['task']
if 'primary_metric' not in task:
    raise KeyError(f"Missing _protocol['task']['primary_metric']; available: {list(task)}")
metric = task['primary_metric']

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

with open(inp['05_preprocess_fit_preprocessing_T5EPBF1H/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

def input_columns(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return None

feat = input_columns(pre)
if feat is None:
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    feat = [c for c in train.columns if c not in drop]
    n = len(train)
    keep = []
    for c in feat:
        s = train[c]
        if s.nunique(dropna=False) == n and (s.dtype.kind in 'iu' or s.dtype == object):
            continue
        keep.append(c)
    feat = keep

forbidden = set(feat) & (set(targets) | set(excluded) | ({group} if group else set()))
if forbidden:
    raise ValueError(f"Feature list contains forbidden columns: {forbidden}")
missing = [c for c in feat if c not in train.columns or c not in val.columns]
if missing:
    raise ValueError(f"Missing feature columns in data: {missing}")

Xtr = train[feat].copy()
ytr = train[targets].astype(float).copy()
Xva = val[feat].copy()
yva = val[targets].astype(float).copy()

model = Pipeline([('preprocess', pre), ('model', LinearRegression())])
model = mlca_runtime.fit_record(model, Xtr, 'train', ytr)

pred = np.asarray(model.predict(Xva))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
ytrue = yva.to_numpy()
rmses = []
for i, target in enumerate(targets):
    rmse = float(np.sqrt(mean_squared_error(ytrue[:, i], pred[:, i])))
    rmses.append(rmse)
    mlca_runtime.record_metric(f"{metric}_{target}", rmse, split='val')
mean_rmse = float(np.mean(rmses))
mlca_runtime.record_metric(f"{metric}_mean", mean_rmse, split='val')
mlca_runtime.save_model(model, "model.pkl")

print(f"features: {len(feat)}")
print(f"train rows: {len(Xtr)}")
print(f"val {metric} mean: {mean_rmse:.6g}")
