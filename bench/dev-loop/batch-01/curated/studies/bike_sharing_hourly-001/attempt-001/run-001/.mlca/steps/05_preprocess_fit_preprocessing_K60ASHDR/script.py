import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
for k in ("targets", "excluded_columns", "group_column", "task"):
    if k not in proto:
        raise KeyError(f"missing {k}; available: {list(proto.keys())}")

targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group = proto["group_column"]

train = pd.read_parquet(inputs["train"])
n = len(train)

feat_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == n:
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == n:
        continue
    feat_cols.append(c)

y = train[targets[0]] if targets else None

num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(train[c]) and (train[c].dropna() >= 0).all()]
derived = set()
for i, a in enumerate(num_cols):
    if train[a].equals(y):
        derived.add(a)
    for j in range(i + 1, len(num_cols)):
        b = num_cols[j]
        if (train[a] + train[b]).equals(y):
            derived.add(a); derived.add(b)
        for k in range(j + 1, len(num_cols)):
            c = num_cols[k]
            if (train[a] + train[b] + train[c]).equals(y):
                derived.add(a); derived.add(b); derived.add(c)
for c in derived:
    feat_cols.remove(c)
if derived:
    print(f"excluded derivations: {sorted(derived)}")

warn = []
for c in feat_cols:
    if pd.api.types.is_numeric_dtype(train[c]):
        r = abs(train[c].corr(y))
        if r >= 0.98:
            warn.append((c, r))
if warn:
    print("warning |corr|>=0.98: " + ", ".join(f"{c}({r:.3f})" for c, r in warn))

placeholders = {'?', '-', 'NA', 'N/A', ''}
ph_cols = []
for c in feat_cols:
    s = train[c]
    if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s):
        vals = set(s.dropna().astype(str).str.strip().unique())
        if vals & placeholders:
            ph_cols.append(c)
if ph_cols:
    print(f"placeholders in {len(ph_cols)} columns")

def clean_text(X):
    X = X.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_datetime64_any_dtype(X[c]):
            X[c] = X[c].astype(str).str.strip().replace(
                {'?': np.nan, '-': np.nan, 'NA': np.nan, 'N/A': np.nan, '': np.nan})
    return X

def make_date_features(X):
    X = X.copy()
    for c in X.columns:
        dt = pd.to_datetime(X[c])
        X[c + '_hour'] = dt.dt.hour
        X[c + '_dow'] = dt.dt.dayofweek
        X[c + '_month'] = dt.dt.month
        X = X.drop(columns=[c])
    return X

threshold = 0.05 * n
num_features, cat_features, date_features = [], [], []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_features.append(c)
    elif pd.api.types.is_datetime64_any_dtype(s):
        date_features.append(c)
    else:
        try:
            pd.to_datetime(s.dropna().head(100), errors='raise')
            date_features.append(c)
        except Exception:
            if s.nunique(dropna=True) <= threshold:
                cat_features.append(c)

transformers = []
if date_features:
    transformers.append(('date', FunctionTransformer(make_date_features, validate=False), date_features))
if num_features:
    transformers.append(('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                                          ('sc', StandardScaler())]), num_features))
if cat_features:
    transformers.append(('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                                          ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
                         cat_features))

ct = ColumnTransformer(transformers, remainder='drop')
ct.set_output(transform='pandas')

preprocessor = Pipeline([('clean', FunctionTransformer(clean_text, validate=False)), ('ct', ct)])
preprocessor.set_output(transform='pandas')

X_train = train[feat_cols]
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, "train")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
print(f"fitted preprocessor: {X_train.shape[0]} rows, {X_train.shape[1]} features")
