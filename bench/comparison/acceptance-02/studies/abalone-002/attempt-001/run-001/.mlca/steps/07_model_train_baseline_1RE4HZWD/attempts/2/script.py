import mlca_runtime, json, pickle
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = proto['targets']
excluded = proto.get('excluded_columns', [])
group = proto.get('group_column')
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val']) if 'val' in inputs else None
feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
feature_cols = [c for c in feature_cols if not (train[c].dtype == object and train[c].nunique() == len(train))]
X_train = train[feature_cols].copy()
y_train = train[targets].copy()

pre = sel = None
pre_path = inputs.get('05_preprocess_fit_preprocessing_NRNQX0TZ/preprocessor.pkl')
sel_path = inputs.get('06_features_retain_features_YTKXSJV5/selector_all.pkl')
if pre_path and Path(pre_path).exists():
    with open(pre_path, 'rb') as f: pre = pickle.load(f)
if sel_path and Path(sel_path).exists():
    with open(sel_path, 'rb') as f: sel = pickle.load(f)

pipe = None
if pre is not None and sel is not None:
    try:
        sample = X_train.head(5)
        pre_out = pre.transform(sample)
        if hasattr(pre_out, 'columns'):
            sel_in = getattr(sel, 'feature_names_in_', None)
            if sel_in is None or list(sel_in) == list(pre_out.columns):
                sel.transform(pre_out)
                pipe = Pipeline([('preprocessor', pre), ('selector', sel), ('model', LinearRegression())])
    except Exception:
        pass
if pipe is None and sel is not None:
    try:
        sel.transform(X_train.head(5))
        if isinstance(sel, Pipeline):
            steps = list(sel.steps)
            if hasattr(steps[-1][1], 'fit') and not isinstance(steps[-1][1], LinearRegression):
                if steps[-1][1].__class__.__name__.endswith(('Regressor', 'Classifier')):
                    steps[-1] = ('model', LinearRegression())
                else:
                    steps = steps + [('model', LinearRegression())]
                pipe = Pipeline(steps)
            else:
                pipe = sel
        else:
            pipe = Pipeline([('selector', sel), ('model', LinearRegression())])
    except Exception:
        pass
if pipe is None and pre is not None:
    try:
        pre.transform(X_train.head(5))
        pipe = Pipeline([('preprocessor', pre), ('model', LinearRegression())])
    except Exception:
        pass
if pipe is None:
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    pre_fresh = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
    pre_fresh.set_output(transform='pandas')
    pipe = Pipeline([('preprocessor', pre_fresh), ('model', LinearRegression())])

fitted = mlca_runtime.fit_record(pipe, X_train, 'train', y=y_train)

if val is not None:
    X_val = val[feature_cols].copy()
    y_val = val[targets].copy()
    preds = np.asarray(fitted.predict(X_val))
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    rmses = [float(np.sqrt(mean_squared_error(y_val[t], preds[:, i]))) for i, t in enumerate(targets)]
    for i, t in enumerate(targets):
        mlca_runtime.record_metric(f'rmse_{t}', rmses[i], 'val')
    mlca_runtime.record_metric('rmse_mean', float(np.mean(rmses)), 'val')
    print(f"Val RMSE mean: {np.mean(rmses):.4f}")
mlca_runtime.save_model(fitted, "model.pkl")
print("Saved model.pkl")
print(f"Features: {len(feature_cols)}")
