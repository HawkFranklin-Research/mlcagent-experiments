import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp['train'])
protocol = json.loads(inp['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns') or []
group_col = protocol.get('group_column')

with open(inp['05_preprocess_fit_preprocessing_DXHK0F72/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = list(preprocessor.feature_names_in_)
else:
    skip = set(targets) | set(excluded)
    if group_col:
        skip.add(group_col)
    n = len(train)
    feature_cols = []
    for c in train.columns:
        if c in skip:
            continue
        s = train[c]
        is_key = (n > 1 and s.is_unique and
                  (s.dtype == object or pd.api.types.is_string_dtype(s)
                   or pd.api.types.is_integer_dtype(s)))
        if is_key:
            continue
        feature_cols.append(c)

missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise ValueError(f"missing feature columns: {missing}")

X_train = train[feature_cols]

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', FunctionTransformer(validate=False, feature_names_out='one-to-one')),
])

fitted = mlca_runtime.fit_record(pipe, X_train, 'train')

n_in = len(feature_cols)
try:
    n_out = len(fitted.get_feature_names_out())
except Exception:
    n_out = n_in
print(f"features in={n_in} out={n_out} rows={len(X_train)}")
mlca_runtime.record_metric('n_features_in', float(n_in), 'train')
mlca_runtime.record_metric('n_features_out', float(n_out), 'train')
mlca_runtime.record_metric('n_train_rows', float(len(X_train)), 'train')

mlca_runtime.save_model(fitted, 'selector_all.pkl')
