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
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])

def key_or_raise(d, k):
    if k not in d:
        raise KeyError(f"Missing '{k}'. Available: {list(d.keys())}")
    return d[k]

targets = key_or_raise(protocol, "targets")
excluded = key_or_raise(protocol, "excluded_columns")
group = protocol.get("group_column")
task = key_or_raise(protocol, "task")

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feature_cols = [c for c in train.columns if c not in drop]

key_like = []
for c in feature_cols:
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        if s.nunique() == len(train):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique() == len(train):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

excluded_feats = set()
for t in targets:
    if t not in train: continue
    y = train[t]
    if not pd.api.types.is_numeric_dtype(y): continue
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train[c]) and train[c].equals(y):
            excluded_feats.add(c)
feature_cols = [c for c in feature_cols if c not in excluded_feats]
if excluded_feats:
    print(f"Excluded derivations: {sorted(excluded_feats)}")

num_targets = [t for t in targets if t in train and pd.api.types.is_numeric_dtype(train[t])]
if num_targets:
    warn = []
    for t in num_targets:
        y = train[t]
        for c in feature_cols:
            if pd.api.types.is_numeric_dtype(train[c]):
                corr = np.corrcoef(train[c].fillna(0), y.fillna(0))[0,1]
                if abs(corr) >= 0.98:
                    warn.append(c)
    if warn:
        print(f"High correlation with target: {sorted(set(warn))}")

placeholders = ['?', '-', 'NA', 'N/A', '', ' ']
ph_cols = []
for c in feature_cols:
    if train[c].dtype == object or pd.api.types.is_string_dtype(train[c]):
        s = train[c].astype(str).str.strip()
        if s.isin(placeholders).any():
            ph_cols.append(c)
if ph_cols:
    print(f"Placeholders found in columns: {ph_cols}")
    for c in ph_cols:
        train[c] = train[c].astype(str).str.strip().replace(placeholders, np.nan)

num_cols, cat_cols, date_cols = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        if s.nunique() > 0.05 * len(train):
            date_cols.append(c)
        else:
            cat_cols.append(c)
if date_cols:
    print(f"Dropped datetime/high-card text columns: {date_cols}")

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])
preprocessor.set_output(transform="pandas")

X_train = train[num_cols + cat_cols].copy()
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, partition="train")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
print(f"Fitted preprocessor on {X_train.shape[1]} features ({len(num_cols)} numeric, {len(cat_cols)} categorical).")
