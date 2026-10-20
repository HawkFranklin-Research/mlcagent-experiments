import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths['_protocol'].read_text())
targets = proto['targets']
excluded = proto.get('excluded_columns', [])
group_col = proto.get('group_column', None)

train = pd.read_parquet(paths['train'])
val = pd.read_parquet(paths['val'])

with open(paths['05_preprocess_fit_preprocessing_XDGKSNK5/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(paths['06_features_retain_features_W9C3S8F0/selector_all.pkl'], 'rb') as f:
    sel = pickle.load(f)


def get_in_cols(obj):
    if obj is None:
        return None
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return None


raw_cols = set(train.columns)
sel_cols = get_in_cols(sel)
if sel_cols is not None and set(sel_cols).issubset(raw_cols):
    frontend = sel
else:
    frontend = Pipeline([('pre', pre), ('sel', sel)]) if sel is not None else pre

feat_cols = get_in_cols(frontend)
if feat_cols is None:
    feat_cols = [c for c in train.columns
                 if c not in targets and c not in excluded and c != group_col]

X = train[feat_cols]
y = train[targets]

pipe = Pipeline([
    ('frontend', frontend),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42)),
])
mlca_runtime.fit_record(pipe, X, 'train', y)
mlca_runtime.save_model(pipe, 'alternative_model.pkl')

pred = np.asarray(pipe.predict(val[feat_cols]))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
y_val = val[targets]
for i, t in enumerate(targets):
    rmse = float(np.sqrt(np.mean((pred[:, i] - y_val[t].values) ** 2)))
    mlca_runtime.record_metric(f'rmse_{t}', rmse, 'val')
    print(f'val {t} rmse: {rmse:.4f}')
print(f'train rows: {len(train)}, features: {len(feat_cols)}, targets: {targets}')
