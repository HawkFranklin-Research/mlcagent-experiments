import mlca_runtime
import json, pickle
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(Path(inputs['_protocol']).read_text())
for k in ('targets', 'task', 'excluded_columns'):
    if k not in protocol:
        raise ValueError(f"Missing protocol[{k!r}]. Available keys: {list(protocol.keys())}")
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])

pre = None
try:
    with open(inputs['05_preprocess_fit_preprocessing_KCWWRC66/preprocessor.pkl'], 'rb') as f:
        pre = pickle.load(f)
except Exception:
    pre = None

if isinstance(pre, Pipeline) and hasattr(pre, 'feature_names_in_'):
    base = pre
    input_cols = list(pre.feature_names_in_)
else:
    drop = set(targets) | set(excluded) | ({group} if group else set())
    num_cols, cat_cols = [], []
    n = len(train)
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == n:
                continue
            num_cols.append(c)
        else:
            if s.nunique(dropna=True) > 0.05 * n:
                continue
            vals = s.dropna().unique()
            if len(vals) and set(map(str, vals)).issubset({'0', '1', '0.0', '1.0', 'True', 'False'}):
                num_cols.append(c)
            else:
                cat_cols.append(c)
    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    cat_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    base = ColumnTransformer(
        [('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)],
        remainder='drop',
    )
    base.set_output(transform='pandas')
    input_cols = num_cols + cat_cols

selector = FunctionTransformer()
pipeline = Pipeline([('pre', base), ('selector', selector)])
try:
    pipeline.set_output(transform='pandas')
except Exception:
    pass

mlca_runtime.fit_record(pipeline, train[input_cols], 'train')
mlca_runtime.save_model(pipeline, 'selector_all.pkl')
print('saved selector_all.pkl features=', len(input_cols))
