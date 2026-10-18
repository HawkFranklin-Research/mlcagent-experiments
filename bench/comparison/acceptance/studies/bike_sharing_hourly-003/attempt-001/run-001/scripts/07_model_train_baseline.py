import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')
metric = protocol['task']['primary_metric']

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

drop = set(targets) | set(excluded) | ({group} if group else set())
contract_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if len(s) > 1 and s.nunique(dropna=True) == len(s):
        k = s.dtype.kind
        if k in 'OUS' or str(s.dtype) in ('category', 'string', 'object'):
            continue
        if k in 'iu':
            continue
    contract_cols.append(c)

with open(inp['05_preprocess_fit_preprocessing_CK5QBBYX/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inp['06_features_retain_features_5J0R0BF8/selector_all.pkl'], 'rb') as f:
    sel = pickle.load(f)

if hasattr(pre, 'feature_names_in_') and set(pre.feature_names_in_) == set(contract_cols):
    feature_cols = list(pre.feature_names_in_)
else:
    feature_cols = contract_cols
assert not (set(feature_cols) & drop)

X, y = train[feature_cols], train[targets]
Xv, yv = val[feature_cols], val[targets]

for obj in (pre, sel):
    try:
        obj.set_output(transform='pandas')
    except Exception:
        pass

def has_pre(obj):
    if isinstance(obj, ColumnTransformer):
        return True
    if isinstance(obj, Pipeline):
        return any(isinstance(s, ColumnTransformer) for s in obj.named_steps.values())
    return False

use_sel = has_pre(sel)
if not use_sel and hasattr(sel, 'feature_names_in_'):
    use_sel = set(sel.feature_names_in_) == set(feature_cols)

fe = sel if use_sel else Pipeline([('preprocessor', pre), ('selector', sel)])
pipe = Pipeline([('fe', fe), ('reg', LinearRegression())])
fitted = mlca_runtime.fit_record(pipe, X, 'train', y=y)
mlca_runtime.save_model(fitted, 'model.pkl')

pred = fitted.predict(Xv)
if len(targets) == 1:
    headline = float(np.sqrt(np.mean((pred.ravel() - yv.values.ravel()) ** 2)))
    mlca_runtime.record_metric(metric, headline, 'val')
else:
    rms = [float(np.sqrt(np.mean((pred[:, i] - yv.values[:, i]) ** 2))) for i in range(len(targets))]
    for t, r in zip(targets, rms):
        mlca_runtime.record_metric(f'{metric}_{t}', r, 'val')
    headline = float(np.mean(rms))
    mlca_runtime.record_metric(metric, headline, 'val')

print(f"train={X.shape} val={Xv.shape} features={len(feature_cols)}")
print(f"val_{metric}={headline:.4f}")
