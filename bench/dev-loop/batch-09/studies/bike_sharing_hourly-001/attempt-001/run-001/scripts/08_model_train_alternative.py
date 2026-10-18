import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_poisson_deviance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

def getp(k):
    if k not in protocol:
        raise KeyError(f"Missing {k}; available: {list(protocol)}")
    return protocol[k]

targets = getp('targets')
excluded = getp('excluded_columns')
group_col = protocol.get('group_column')
task = getp('task')
primary_metric = task['primary_metric']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

def is_key_like(s):
    if s.nunique(dropna=False) != len(s) or pd.api.types.is_float_dtype(s):
        return False
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s))

candidate = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    if is_key_like(train[c]):
        continue
    candidate.append(c)

with open(inputs['05_preprocess_fit_preprocessing_Q23GTBD5/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
pre_steps = list(pre.named_steps.keys()) if isinstance(pre, Pipeline) else []

if hasattr(pre, 'feature_names_in_'):
    expected = list(pre.feature_names_in_)
    feature_cols = expected if set(expected).issubset(set(candidate)) else candidate
else:
    feature_cols = candidate

X_train = train[feature_cols].copy()
y_train = train[targets].copy()
X_val = val[feature_cols].copy()
y_val = val[targets].copy()

use_pre = pre is not None and not any(s in pre_steps for s in ['model', 'estimator', 'regressor', 'classifier'])
if use_pre and hasattr(pre, 'feature_names_in_'):
    use_pre = set(pre.feature_names_in_).issubset(set(feature_cols))

if use_pre:
    steps = [('preprocessor', pre)]
else:
    for c in feature_cols:
        if pd.api.types.is_datetime64_any_dtype(X_train[c]):
            X_train[c] = X_train[c].astype('int64') // 10**9
            X_val[c] = X_val[c].astype('int64') // 10**9
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_train[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    ct = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)], remainder='drop')
    ct.set_output(transform='pandas')
    steps = [('preprocessor', ct)]

steps.append(('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)))
pipeline = Pipeline(steps)
fitted = mlca_runtime.fit_record(pipeline, X_train, 'train', y=y_train)

pred = fitted.predict(X_val)
pred = np.clip(pred, 0, None)
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)

scores = []
for i, t in enumerate(targets):
    score = mean_poisson_deviance(y_val[t].values, pred[:, i])
    scores.append(score)
    mlca_runtime.record_metric(f'poisson_deviance_{t}', score, 'val')

headline = float(np.mean(scores))
mlca_runtime.record_metric('poisson_deviance', headline, 'val')
mlca_runtime.save_model(fitted, 'alternative_model.pkl')
print(f"features: {len(feature_cols)}")
print(f"val poisson_deviance: {headline:.4f}")
