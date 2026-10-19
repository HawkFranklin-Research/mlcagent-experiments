import mlca_runtime, json
import pandas as pd, numpy as np
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
feature_cols = [c for c in feature_cols
                if not (not pd.api.types.is_numeric_dtype(train[c])
                        and not pd.api.types.is_datetime64_any_dtype(train[c])
                        and train[c].nunique() > max(5, int(0.05 * len(train))))]

X_train = train[feature_cols].copy()
y_train = train[targets].copy()
mask = y_train.notna().all(axis=1)
X_train = X_train.loc[mask].copy()
y_train = y_train.loc[mask].copy()

num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]

num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler())])
cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                     ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
pre = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
pre.set_output(transform='pandas')

pipe = Pipeline([('preprocessor', pre), ('model', LinearRegression())])
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
print(f"Features: {len(feature_cols)}  num={len(num_cols)}  cat={len(cat_cols)}")
