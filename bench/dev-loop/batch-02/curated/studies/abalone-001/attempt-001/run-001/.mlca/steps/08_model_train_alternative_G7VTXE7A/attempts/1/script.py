import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

mlca_runtime.set_seed(42)

def get_protocol_key(proto, key):
    if key not in proto:
        raise KeyError(f"Missing key '{key}' in _protocol. Available keys: {list(proto.keys())}")
    return proto[key]

inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = get_protocol_key(proto, "targets")
excluded = get_protocol_key(proto, "excluded_columns")
group_col = get_protocol_key(proto, "group_column")
_ = get_protocol_key(proto, "task")

train = pd.read_parquet(inputs["train"])

def compute_feature_cols(df, targets, excluded, group_col):
    cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]
    key_like = []
    for c in cols:
        s = df[c]
        if s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s.dtype):
            if s.nunique(dropna=False) == len(s):
                key_like.append(c)
        elif pd.api.types.is_integer_dtype(s.dtype):
            if s.nunique(dropna=False) == len(s):
                key_like.append(c)
    return [c for c in cols if c not in key_like]

raw_cols = compute_feature_cols(train, targets, excluded, group_col)

with open(inputs["05_preprocess_fit_preprocessing_JT5JG4F1/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_retain_features_6CDSH3YD/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)

def wraps_preproc(sel, raw_cols):
    if not isinstance(sel, Pipeline):
        return False
    if not hasattr(sel, 'feature_names_in_'):
        return False
    return set(sel.feature_names_in_).issubset(set(raw_cols))

if selector is not None and wraps_preproc(selector, raw_cols):
    frontend = selector
elif preprocessor is not None and selector is not None:
    frontend = Pipeline([('preprocessor', preprocessor), ('selector', selector)])
elif preprocessor is not None:
    frontend = preprocessor
elif selector is not None:
    frontend = selector
else:
    frontend = None

if frontend is not None and hasattr(frontend, 'feature_names_in_'):
    feature_cols = list(frontend.feature_names_in_)
else:
    feature_cols = raw_cols

if frontend is None:
    num_cols = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    ct = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
    ct.set_output(transform='pandas')
    frontend = ct

pipeline = Pipeline([
    ('frontend', frontend),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

X = train[feature_cols]
y = train[targets]

mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(pipeline, "alternative_model.pkl")

print(f"Train shape: {train.shape}")
print(f"Features used: {len(feature_cols)}")
print(f"Targets: {targets}")
print("Saved alternative_model.pkl")
