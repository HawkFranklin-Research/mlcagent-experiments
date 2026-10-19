import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])

missing = [k for k in ['targets', 'excluded_columns', 'task'] if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol.keys())}")
targets = list(protocol['targets'])
excluded = list(protocol['excluded_columns'])
group = protocol.get('group_column')

drop_cols = set(targets) | set(excluded)
if group:
    drop_cols.add(group)

n = len(train)

def is_keylike(s):
    if s.nunique(dropna=False) != n:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) > 2:
        return True
    return False

feature_cols = [c for c in train.columns if c not in drop_cols and not is_keylike(train[c])]
train_X = train[feature_cols]
train_y = train[targets]

ph = {'?', '-', 'NA', 'N/A', ''}
ph_cols = 0
for c in feature_cols:
    s = train_X[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        st = s.astype(str).str.strip()
        if (st.isin(ph) & (st != '')).any():
            ph_cols += 1
print(f"placeholder_columns={ph_cols}")

to_drop = set()
for c in feature_cols:
    if not pd.api.types.is_numeric_dtype(train_X[c]):
        continue
    v = train_X[c].values.astype(float)
    for t in targets:
        if pd.api.types.is_numeric_dtype(train_y[t]) and np.allclose(
            v, train_y[t].values.astype(float), equal_nan=True, rtol=1e-9
        ):
            to_drop.add(c)
            print(f"excluded_derived:{c}=={t}")
            break

warn = []
for c in feature_cols:
    if c in to_drop or not pd.api.types.is_numeric_dtype(train_X[c]):
        continue
    for t in targets:
        if not pd.api.types.is_numeric_dtype(train_y[t]):
            continue
        try:
            pair = pd.concat([train_X[c], train_y[t]], axis=1).dropna()
            corr = float(pair.corr().iloc[0, 1])
            if not np.isnan(corr) and abs(corr) >= 0.98:
                warn.append(c)
                break
        except Exception:
            pass
if warn:
    print(f"corr_warning={warn}")

final_features = [c for c in feature_cols if c not in to_drop]
num_cols, cat_cols = [], []
for c in final_features:
    s = train[c]
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique() <= max(5, int(0.05 * n)):
            cat_cols.append(c)
    else:
        num_cols.append(c)

num_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
])
num_pipe.set_output(transform='pandas')
parts = []
if num_cols:
    parts.append(('num', num_pipe, num_cols))
if cat_cols:
    cat_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    cat_pipe.set_output(transform='pandas')
    parts.append(('cat', cat_pipe, cat_cols))

pre = ColumnTransformer(parts)
pre.set_output(transform='pandas')

fitted = mlca_runtime.fit_record(pre, train[final_features], 'train')
mlca_runtime.save_model(fitted, 'preprocessor.pkl')
nout = fitted.transform(train[final_features]).shape[1]
print(f"n_in={len(final_features)} n_out={nout}")
