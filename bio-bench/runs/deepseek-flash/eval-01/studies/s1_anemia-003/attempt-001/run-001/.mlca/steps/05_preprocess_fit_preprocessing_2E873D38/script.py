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
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)

drop_cols = set(targets) | set(excluded)
if group_col:
    drop_cols.add(group_col)

# key-like columns: object with all unique; integer contiguous sequence
for c in train.columns:
    if c in drop_cols:
        continue
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        if train[c].nunique() == len(train):
            drop_cols.add(c)
    elif pd.api.types.is_integer_dtype(train[c]) and train[c].nunique() == len(train):
        vals = train[c].dropna().unique()
        if len(vals) == len(train) and (vals.max() - vals.min() + 1) == len(vals):
            drop_cols.add(c)

feature_cols = [c for c in train.columns if c not in drop_cols]

# convert datetime columns to int64 (nanoseconds)
for c in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(train[c]):
        train[c] = train[c].astype('int64')
        val[c] = val[c].astype('int64')

# screen target derivations on training rows
y_train = train[targets[0]] if len(targets) == 1 else None
derived = []
if y_train is not None and y_train.dtype.kind in 'if':
    for c in feature_cols:
        if train[c].dtype.kind in 'if' and train[c].equals(y_train):
            derived.append((c, 'exact equality'))
    num_feats = [c for c in feature_cols if train[c].dtype.kind in 'if' and (train[c] >= 0).all()]
    for i in range(len(num_feats)):
        for j in range(i+1, len(num_feats)):
            s = train[num_feats[i]] + train[num_feats[j]]
            if s.equals(y_train):
                derived.append((f"{num_feats[i]}+{num_feats[j]}", 'sum equals target'))
            for k in range(j+1, len(num_feats)):
                s3 = s + train[num_feats[k]]
                if s3.equals(y_train):
                    derived.append((f"{num_feats[i]}+{num_feats[j]}+{num_feats[k]}", 'sum equals target'))
    for c, reason in derived:
        print(f"Excluded derived feature {c} ({reason})")
        if c in feature_cols:
            feature_cols.remove(c)

# correlation warning for numeric binary target
if y_train is not None and y_train.dtype.kind in 'if' and protocol['task']['type'] == 'binary':
    warns = []
    for c in feature_cols:
        if train[c].dtype.kind in 'if':
            corr = train[c].corr(y_train)
            if abs(corr) >= 0.98:
                warns.append(c)
    if warns:
        print(f"High corr with target >=0.98: {', '.join(warns)}")

# placeholder tokens
placeholders = ['?', '-', 'NA', 'N/A', '']
ph_cols = []
for c in feature_cols:
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        s = train[c].astype(str).str.strip()
        if s.isin(placeholders).any():
            ph_cols.append(c)
if ph_cols:
    print(f"Placeholders in {len(ph_cols)} columns: {', '.join(ph_cols)}")
    for c in ph_cols:
        train[c] = train[c].astype(str).str.strip().replace(placeholders, np.nan)
        val[c] = val[c].astype(str).str.strip().replace(placeholders, np.nan)

num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train[c])]

# drop high-cardinality categorical (>5% unique)
n = len(train)
for c in list(cat_cols):
    if train[c].nunique() > 0.05 * n:
        print(f"Dropping high-cardinality categorical {c}")
        cat_cols.remove(c)
        feature_cols.remove(c)

transformers = []
if num_cols:
    transformers.append(('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_cols))
if cat_cols:
    transformers.append(('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cat_cols))

preprocessor = ColumnTransformer(transformers, remainder='drop')
width = len(num_cols) + sum(train[c].nunique() for c in cat_cols)
if width <= 300:
    preprocessor.set_output(transform='pandas')
else:
    for name, trans, _ in preprocessor.transformers:
        if name == 'cat':
            for step_name, step in trans.steps:
                if step_name == 'onehot':
                    step.set_params(sparse_output=True)

X_train = train[feature_cols]
mlca_runtime.fit_record(preprocessor, X_train, 'train', y=y_train)
val_trans = preprocessor.transform(val[feature_cols])
print(f"Val transformed shape: {val_trans.shape}")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
