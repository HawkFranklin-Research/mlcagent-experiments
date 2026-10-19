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

try:
    targets = protocol['targets']
    excluded = protocol['excluded_columns']
except KeyError as e:
    raise ValueError(f"Missing protocol key {e}. Available: {list(protocol.keys())}")
group_col = protocol.get('group_column', None)
task = protocol.get('task', {})
primary_metric = task.get('primary_metric', 'roc_auc_ovr')

candidate_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
feature_cols = []
for c in candidate_cols:
    s = train[c]
    if s.nunique(dropna=False) == len(train):
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            continue
        if pd.api.types.is_integer_dtype(s):
            continue
    feature_cols.append(c)

target = train[targets[0]]
dropped = []
for c in list(feature_cols):
    if train[c].equals(target):
        dropped.append(c)
        feature_cols.remove(c)
for c in dropped:
    print(f"Excluded feature {c}: exact equality with target")

ph_tokens = {'?', '-', 'NA', 'N/A', ''}
ph_cols = []
for c in feature_cols:
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        for df in (train, val):
            if c not in df.columns:
                continue
            s = df[c]
            mask_orig = s.isna()
            s2 = s.astype(str).str.strip()
            mask_ph = s2.isin(ph_tokens) & ~mask_orig
            if mask_ph.any() and df is train:
                ph_cols.append(c)
            s2 = s2.where(~mask_ph, np.nan)
            s2[mask_orig] = np.nan
            df[c] = s2
if ph_cols:
    print(f"Placeholders found in {len(ph_cols)} columns")

numeric_cols = []
cat_cols = []
for c in list(feature_cols):
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        train[c+'_dayofweek'] = s.dt.dayofweek
        val[c+'_dayofweek'] = val[c].dt.dayofweek
        train[c+'_month'] = s.dt.month
        val[c+'_month'] = val[c].dt.month
        numeric_cols.extend([c+'_dayofweek', c+'_month'])
        feature_cols.remove(c)
    elif pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    else:
        n_unique = s.nunique(dropna=True)
        if n_unique <= max(1, int(0.05 * len(train))):
            cat_cols.append(c)
        else:
            feature_cols.remove(c)

feature_cols = numeric_cols + cat_cols

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
ct = ColumnTransformer([
    ('num', num_pipe, numeric_cols),
    ('cat', cat_pipe, cat_cols)
], remainder='drop')

width = len(numeric_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
if width <= 300:
    ct.set_output(transform='pandas')

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()

mlca_runtime.fit_record(ct, X_train, 'train')
X_val_t = ct.transform(X_val)
print(f"Train shape {X_train.shape}, Val transformed shape {X_val_t.shape}")

mlca_runtime.save_model(ct, 'preprocessor.pkl')
mlca_runtime.record_metric('n_features', len(feature_cols), 'train')
mlca_runtime.record_metric('n_train', len(X_train), 'train')
mlca_runtime.record_metric('n_val', len(X_val), 'val')
