import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())

try:
    targets = list(proto['targets'])
    excluded = list(proto['excluded_columns'])
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available top-level keys: {list(proto.keys())}") from e
group = proto.get('group_column', None)

train = pd.read_parquet(inp['train'])
drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == len(s) and len(s) > 1:
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s) and len(s) > 1:
        vals = set(pd.unique(s.dropna()))
        if not vals.issubset({0, 1}):
            continue
    feature_cols.append(c)

X = train[feature_cols].copy()
y = train[targets].copy()

def load_pkl(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

pre = load_pkl(inp['05_preprocess_fit_preprocessing_DXHK0F72/preprocessor.pkl'])
sel = load_pkl(inp['06_features_retain_features_NH52E6T3/selector_all.pkl'])

def input_cols(obj):
    fc = getattr(obj, 'feature_names_in_', None)
    return list(fc) if fc is not None else None

def raw_ok(obj):
    fc = input_cols(obj)
    return fc is not None and all(c in X.columns for c in fc)

def output_is_df(obj):
    try:
        return isinstance(obj.transform(X.iloc[:1]), pd.DataFrame)
    except Exception:
        return False

frontend = None
if sel is not None and raw_ok(sel):
    frontend = sel
    X = X[input_cols(sel)].copy()
elif pre is not None and raw_ok(pre):
    X = X[input_cols(pre)].copy()
    frontend = pre
    if sel is not None:
        try:
            out_names = list(pre.get_feature_names_out())
        except Exception:
            out_names = None
        sel_in = input_cols(sel)
        if out_names is not None and sel_in == out_names and output_is_df(pre):
            frontend = Pipeline([('preprocessor', pre), ('selector', sel)])

if frontend is None:
    num_cols, cat_cols = [], []
    for c in feature_cols:
        s = X[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            X[c + '_ts'] = pd.to_datetime(s).astype('int64') // 10**9
            num_cols.append(c + '_ts')
        elif pd.api.types.is_numeric_dtype(s):
            vals = set(pd.unique(s.dropna()))
            if s.nunique(dropna=True) <= 10 and not vals.issubset({0, 1}):
                cat_cols.append(c)
            else:
                num_cols.append(c)
        else:
            if s.nunique(dropna=True) > max(5, int(0.05 * len(X))):
                continue
            cat_cols.append(c)

    transformers = []
    if num_cols:
        transformers.append((
            'num',
            Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            'cat',
            Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
            cat_cols,
        ))
    ct = ColumnTransformer(transformers, remainder='drop')
    ct.set_output(transform='pandas')
    frontend = Pipeline([('preprocessor', ct)])

pipe = Pipeline([
    ('frontend', frontend),
    ('model', MultiOutputClassifier(LogisticRegression(max_iter=2000))),
])
fitted = mlca_runtime.fit_record(pipe, X, 'train', y=y)
mlca_runtime.save_model(fitted, 'model.pkl')
print('saved model.pkl')
