import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

def get_proto_key(proto, key):
    if key not in proto:
        raise KeyError(f"Missing key '{key}' in _protocol. Available keys: {list(proto.keys())}")
    return proto[key]

targets = get_proto_key(proto, "targets")
excluded = get_proto_key(proto, "excluded_columns")
group_col = get_proto_key(proto, "group_column")
task = get_proto_key(proto, "task")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

candidate_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
key_like = []
for c in candidate_cols:
    if train[c].nunique(dropna=False) == len(train):
        if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]) or pd.api.types.is_integer_dtype(train[c]):
            key_like.append(c)
feature_cols = [c for c in candidate_cols if c not in key_like]

excluded_features = []
warn_features = []
for c in feature_cols:
    if train[c].equals(train[targets[0]]):
        excluded_features.append((c, "exact equality with target"))
    else:
        corr = train[c].corr(train[targets[0]])
        if abs(corr) >= 0.98:
            warn_features.append(c)

feature_cols = [c for c in feature_cols if c not in [e[0] for e in excluded_features]]

numeric, categorical, high_card = [], [], []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train[c]):
        numeric.append(c)
    else:
        if train[c].nunique() > 0.05 * len(train):
            high_card.append(c)
        else:
            categorical.append(c)

placeholders = ["?", "-", "NA", "N/A", ""]
placeholder_cols = []
for c in categorical:
    if train[c].dtype == object or pd.api.types.is_string_dtype(train[c]):
        vals = train[c].astype(str).str.strip()
        if any((vals == p).any() for p in placeholders):
            placeholder_cols.append(c)

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.placeholders_ = ["?", "-", "NA", "N/A", ""]
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            if X[col].dtype == object or pd.api.types.is_string_dtype(X[col]):
                X[col] = X[col].astype(str).str.strip()
                X[col] = X[col].replace(self.placeholders_, np.nan)
        return X

estimated_width = len(numeric) + sum(train[c].nunique() for c in categorical)
sparse = estimated_width > 300

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])
cat_pipe = Pipeline([
    ("clean", TextCleaner()),
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, numeric),
    ("cat", cat_pipe, categorical)
], remainder="drop")

if not sparse:
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")
    preprocessor.set_output(transform="pandas")

X_train = train[feature_cols]
X_val = val[feature_cols]
mlca_runtime.fit_record(preprocessor, X_train, "train")
X_val_trans = preprocessor.transform(X_val)

print(f"Features: {len(feature_cols)} (num {len(numeric)}, cat {len(categorical)})")
if excluded_features:
    print(f"Excluded: {excluded_features}")
if high_card or warn_features:
    print(f"High-card: {high_card}; Corr>=0.98: {warn_features}")
if placeholder_cols:
    print(f"Placeholders in {len(placeholder_cols)} cols: {placeholder_cols}")
print(f"Val transformed shape: {X_val_trans.shape}")

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
