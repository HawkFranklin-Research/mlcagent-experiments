import mlca_runtime
import json
import itertools
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
proto = json.loads(inputs["_protocol"].read_text())

targets = proto["targets"]
excluded = proto.get("excluded_columns", [])
group_col = proto.get("group_column", None)

all_cols = list(train.columns)
drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)

n = len(train)
key_like = []
for c in all_cols:
    if c in drop:
        continue
    s = train[c]
    if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
        if s.nunique(dropna=True) == n:
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s.dtype):
        if s.nunique(dropna=True) == n and (s.is_monotonic_increasing or "id" in c.lower()):
            key_like.append(c)
drop |= set(key_like)

features = [c for c in all_cols if c not in drop]

placeholder_tokens = {"?", "-", "NA", "N/A", ""}
text_cols = [
    c for c in features
    if pd.api.types.is_string_dtype(train[c].dtype) or train[c].dtype == object
]
placeholder_cols = []
for c in text_cols:
    s = train[c].astype("string").str.strip()
    if s.isin(placeholder_tokens).any():
        placeholder_cols.append(c)
print(f"placeholder tokens {sorted(placeholder_tokens)} in {len(placeholder_cols)} columns")

y_train = train[targets[0]]
derived = set()
for c in features:
    if pd.api.types.is_numeric_dtype(train[c]) and pd.api.types.is_numeric_dtype(y_train):
        if np.allclose(
            train[c].astype(float).values,
            y_train.astype(float).values,
            equal_nan=False,
        ):
            derived.add(c)

num_features = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
nonneg = [c for c in num_features if (train[c] >= 0).all()]
for r in (1, 2, 3):
    for combo in itertools.combinations(nonneg, r):
        s = train[list(combo)].sum(axis=1)
        if np.allclose(s.values, y_train.values, equal_nan=False):
            derived.update(combo)
features = [c for c in features if c not in derived]
print(f"excluded derived features: {sorted(derived) if derived else 'none'}")

numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
categorical_features = [c for c in features if c not in numeric_features]
high_card = [
    c for c in categorical_features
    if train[c].nunique(dropna=True) > 0.05 * n
]
if high_card:
    features = [c for c in features if c not in high_card]
    numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    categorical_features = [c for c in features if c not in numeric_features]
    print(f"dropped high-cardinality categorical: {high_card}")

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

transformers = []
if numeric_features:
    transformers.append(("num", num_pipe, numeric_features))
if categorical_features:
    transformers.append(("cat", cat_pipe, categorical_features))

preprocessor = ColumnTransformer(transformers)
preprocessor.set_output(transform="pandas")

X_train = train[features]
X_val = val[features]
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, "train", y=y_train)
val_trans = preprocessor.transform(X_val)
print(f"val transformed shape: {val_trans.shape}")

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
