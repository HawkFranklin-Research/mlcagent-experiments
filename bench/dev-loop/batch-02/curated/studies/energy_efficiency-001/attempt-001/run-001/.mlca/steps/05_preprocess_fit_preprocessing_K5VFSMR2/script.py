import mlca_runtime
import pandas as pd
import numpy as np
import json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
train_df = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]
task = protocol["task"]

all_cols = list(train_df.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]

key_like = []
for c in feature_cols:
    s = train_df[c]
    is_text = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)
    is_int = pd.api.types.is_integer_dtype(s)
    if is_text and s.nunique() == len(train_df):
        key_like.append(c)
    elif is_int and not set(s.dropna().unique()).issubset({0, 1}) and s.nunique() == len(train_df):
        key_like.append(c)

feature_cols = [c for c in feature_cols if c not in key_like]
if key_like:
    print(f"Excluded key-like features: {key_like}")

derived = []
for c in feature_cols:
    for t in targets:
        if train_df[c].equals(train_df[t]):
            derived.append((c, t))
            break
feature_cols = [c for c in feature_cols if c not in [d[0] for d in derived]]
if derived:
    print(f"Excluded derivations: {derived}")

warn = []
for t in targets:
    if pd.api.types.is_numeric_dtype(train_df[t]):
        for c in feature_cols:
            if pd.api.types.is_numeric_dtype(train_df[c]):
                corr = train_df[[c, t]].corr().iloc[0, 1]
                if abs(corr) >= 0.98:
                    warn.append(c)
if warn:
    print(f"Correlation warnings: {sorted(set(warn))}")

placeholders = {'?', '-', 'NA', 'N/A', ''}
ph_cols = []
for c in feature_cols:
    s = train_df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        stripped = s.astype(str).str.strip()
        if stripped.isin(placeholders).any():
            ph_cols.append(c)
if ph_cols:
    print(f"Placeholders found in columns: {ph_cols}")
    for c in ph_cols:
        s = train_df[c].astype(str).str.strip()
        s = s.replace({p: np.nan for p in placeholders if p != ''})
        s = s.replace('', np.nan)
        train_df[c] = s

num_cols, cat_cols, drop_cols, derived_cols = [], [], [], []
for c in feature_cols:
    s = train_df[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        sample = s.dropna().astype(str).head(100)
        if len(sample) > 0:
            dt = pd.to_datetime(sample, errors='coerce')
            if dt.notna().mean() > 0.9:
                dt_full = pd.to_datetime(s, errors='coerce')
                for part, func in [('hour', dt_full.dt.hour), ('dow', dt_full.dt.dayofweek), ('month', dt_full.dt.month)]:
                    name = f"{c}_{part}"
                    train_df[name] = func
                    num_cols.append(name)
                    derived_cols.append(name)
                drop_cols.append(c)
                continue
        if s.nunique() <= 0.05 * len(train_df):
            cat_cols.append(c)
        else:
            drop_cols.append(c)
    else:
        if set(s.dropna().unique()).issubset({0, 1}):
            num_cols.append(c)
        else:
            drop_cols.append(c)

feature_cols = [c for c in feature_cols if c not in drop_cols] + derived_cols
if drop_cols:
    print(f"Dropped high-cardinality/date columns: {drop_cols}")

est_width = len(num_cols) + sum(train_df[c].nunique() for c in cat_cols)
use_pandas = est_width <= 300
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=not use_pandas))
])
transformers = []
if num_cols:
    transformers.append(('num', num_pipe, num_cols))
if cat_cols:
    transformers.append(('cat', cat_pipe, cat_cols))
preprocessor = ColumnTransformer(transformers)
if use_pandas:
    num_pipe.set_output(transform='pandas')
    cat_pipe.set_output(transform='pandas')
    preprocessor.set_output(transform='pandas')

X_train = train_df[feature_cols]
mlca_runtime.fit_record(preprocessor, X_train, partition='train')
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
