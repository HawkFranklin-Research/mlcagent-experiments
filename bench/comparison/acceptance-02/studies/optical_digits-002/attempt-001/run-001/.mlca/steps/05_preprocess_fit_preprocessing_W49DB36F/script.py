import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp['train'])
protocol = json.loads(inp['_protocol'].read_text())

targets = list(protocol.get('targets') or [])
excluded = list(protocol.get('excluded_columns') or [])
group = protocol.get('group_column')
drop = set(targets) | set(excluded)
if group:
    drop.add(group)

feat_cols = [c for c in train.columns if c not in drop]
n = len(train)
num_cols, cat_cols, dt_cols = [], [], []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dt_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    else:
        k = s.nunique(dropna=True)
        if k <= max(5, int(0.05 * n)):
            cat_cols.append(c)

num_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc', StandardScaler()),
])
parts = [('num', num_pipe, num_cols)]
if cat_cols:
    cat_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    parts.append(('cat', cat_pipe, cat_cols))

pre = ColumnTransformer(parts, remainder='drop')
pre.set_output(transform='pandas')
X = train[feat_cols]
fitted = mlca_runtime.fit_record(pre, X, 'train')
mlca_runtime.save_model(fitted, 'preprocessor.pkl')
print(f"train rows={n} features={len(feat_cols)} num={len(num_cols)} cat={len(cat_cols)} dt={len(dt_cols)}")
