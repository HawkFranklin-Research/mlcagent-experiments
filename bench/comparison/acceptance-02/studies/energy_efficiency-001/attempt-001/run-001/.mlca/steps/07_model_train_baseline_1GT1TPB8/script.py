import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', []) or []
group_col = protocol.get('group_column', None)

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
n = len(train)

def is_key(s):
    if pd.api.types.is_integer_dtype(s):
        return s.nunique() == n and not s.isin([0, 1]).all()
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique() == n
    return False

feature_cols = [c for c in train.columns if c not in drop and not is_key(train[c])]

X_tr = train[feature_cols].copy()
X_va = val[feature_cols].copy()
y_tr = train[targets].copy()
y_va = val[targets].copy()

with open(inp['05_preprocess_fit_preprocessing_RHXAHV44/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inp['06_features_retain_features_W82TE9KX/selector_all.pkl'], 'rb') as f:
    sel = pickle.load(f)

def accepts_raw(obj, cols):
    fn = getattr(obj, 'feature_names_in_', None)
    return fn is not None and list(fn) == list(cols)

if accepts_raw(sel, feature_cols):
    frontend = sel
else:
    frontend = Pipeline([('pre', pre), ('sel', sel)])

model = Pipeline([('frontend', frontend), ('lr', LinearRegression())])
mlca_runtime.fit_record(model, X_tr, 'train', y=y_tr)

pred = np.asarray(model.predict(X_va))
rmses = [float(np.sqrt(mean_squared_error(y_va.iloc[:, i], pred[:, i])))
         for i in range(len(targets))]
for t, r in zip(targets, rmses):
    mlca_runtime.record_metric(f'rmse_{t}', r, 'val')
mlca_runtime.record_metric('rmse_mean', float(np.mean(rmses)), 'val')
mlca_runtime.save_model(model, 'model.pkl')
print(f'train {X_tr.shape} val {X_va.shape} rmse_mean={np.mean(rmses):.4f}')
