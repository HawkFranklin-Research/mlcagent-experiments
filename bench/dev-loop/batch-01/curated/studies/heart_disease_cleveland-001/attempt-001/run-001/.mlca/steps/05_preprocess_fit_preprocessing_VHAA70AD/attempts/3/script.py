import json
import pandas as pd
import mlca_runtime
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in proto:
        raise KeyError(f"Missing {k} in _protocol. Available: {list(proto.keys())}")
targets = proto["targets"]
excluded = proto["excluded_columns"]
group_col = proto["group_column"]
task = proto["task"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

candidate = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
key_like = []
for c in candidate:
    if train[c].nunique(dropna=False) == len(train):
        if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]) or pd.api.types.is_integer_dtype(train[c]):
            key_like.append(c)
feature_cols = [c for c in candidate if c not in key_like]

excluded_features = []
warn_features = []
for t in targets:
    for c in feature_cols:
        if train[c].equals(train[t]):
            excluded_features.append((c, f"exact equality with {t}"))
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and (train[c] >= 0).all()]
    for r in range(1, 4):
        for combo in combinations(num_cols, r):
            if (train[list(combo)].sum(axis=1) == train[t]).all():
                for c in combo:
                    excluded_features.append((c, f"row sum with {combo} equals {t}"))
excluded_set = set(c for c, _ in excluded_features)
feature_cols = [c for c in feature_cols if c not in excluded_set]

for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train[c]):
        corr = train[c].corr(train[targets[0]])
        if abs(corr) >= 0.98:
            warn_features.append(c)

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
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        s = train[c].astype("string").str.strip()
        if s.isin(placeholders).any():
            placeholder_cols.append(c)

class CleanText(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for c in X.columns:
            if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_string_dtype(X[c]):
                s = X[c].astype("string").str.strip()
                X[c] = s.mask(s.isin(placeholders)).astype(object)
        return X

num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
cat_pipe = Pipeline([("clean", CleanText()), ("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
preprocessor = ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, categorical)], remainder="drop")
num_pipe.set_output(transform="pandas")
cat_pipe.set_output(transform="pandas")
preprocessor.set_output(transform="pandas")

X_train = train[feature_cols]
X_val = val[feature_cols]
mlca_runtime.fit_record(preprocessor, X_train, "train")
X_val_trans = preprocessor.transform(X_val)

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

print(f"Features: {len(feature_cols)} (num {len(numeric)}, cat {len(categorical)})")
if excluded_features:
    print(f"Excluded: {excluded_features}")
if high_card or warn_features:
    print(f"High-card: {high_card}; Corr>=0.98: {warn_features}")
if placeholder_cols:
    print(f"Placeholders in {len(placeholder_cols)} cols: {placeholder_cols}")
print(f"Val transformed shape: {X_val_trans.shape}")
