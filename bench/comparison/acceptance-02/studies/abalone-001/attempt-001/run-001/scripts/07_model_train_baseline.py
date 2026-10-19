import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

mlca_runtime.set_seed(42)
ins = mlca_runtime.inputs()
proto = json.loads(ins['_protocol'].read_text())
targets = list(proto['targets'])
excluded = list(proto.get('excluded_columns') or [])
group = proto.get('group_column', None)

train = pd.read_parquet(ins['train'])
val = pd.read_parquet(ins['val'])
with open(ins['05_preprocess_fit_preprocessing_MZ4P76F5/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(ins['06_features_retain_features_Q9C0R6MS/selector_all.pkl'], 'rb') as f:
    sel = pickle.load(f)

def names_in(obj):
    v = getattr(obj, 'feature_names_in_', None)
    return list(v) if v is not None else None

pre_in = names_in(pre)
if pre_in is not None:
    feat_cols = [c for c in pre_in if c in train.columns and c not in targets
                 and c not in excluded and (group is None or c != group)]
else:
    n = len(train)
    feat_cols = []
    for c in train.columns:
        if c in targets or c in excluded or (group is not None and c == group):
            continue
        s = train[c]
        if pd.api.types.is_integer_dtype(s) and s.nunique() == n:
            continue
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique() == n:
            continue
        feat_cols.append(c)

X = train[feat_cols]
y = train[targets]

def can_raw(obj):
    try:
        obj.transform(X.iloc[:2])
        return True
    except Exception:
        return False

def contains_pre(obj):
    if obj is pre:
        return True
    if isinstance(obj, Pipeline):
        return any(s is pre or isinstance(s, ColumnTransformer) for _, s in obj.steps)
    return False

sel_on_raw = can_raw(sel)
if sel_on_raw and contains_pre(sel):
    base = sel
elif sel_on_raw:
    base = pre
else:
    base = Pipeline([('pre', pre), ('sel', sel)])

steps = list(base.steps) if isinstance(base, Pipeline) else [('base', base)]
pipeline = Pipeline(steps + [('model', LinearRegression())])

model = mlca_runtime.fit_record(pipeline, X, 'train', y=y)
mlca_runtime.save_model(model, 'model.pkl')

pred = model.predict(val[feat_cols])
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
rmses = []
for i, t in enumerate(targets):
    rmse = float(np.sqrt(np.mean((pred[:, i] - val[t].to_numpy()) ** 2)))
    rmses.append(rmse)
    mlca_runtime.record_metric(f'{t}_rmse', rmse, 'val')
mean_rmse = float(np.mean(rmses))
mlca_runtime.record_metric('mean_rmse', mean_rmse, 'val')

print('train', train.shape, 'val', val.shape)
print('features', len(feat_cols), 'targets', targets)
print('val mean_rmse', round(mean_rmse, 4))
