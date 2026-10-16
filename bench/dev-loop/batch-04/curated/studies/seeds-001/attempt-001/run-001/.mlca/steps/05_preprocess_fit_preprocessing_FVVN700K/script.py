import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
train_df = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys: {missing}. Available: {list(protocol.keys())}")

targets = set(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column", None)

feature_cols = [c for c in train_df.columns if c not in targets and c not in excluded and c != group]

key_like = []
for c in list(feature_cols):
    s = train_df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=False) == len(s):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=False) == len(s):
            key_like.append(c)
for c in key_like:
    feature_cols.remove(c)
if key_like:
    print(f"Dropped key-like features: {len(key_like)}")

text_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train_df[c]) or pd.api.types.is_string_dtype(train_df[c])]
placeholders = {"?", "-", "NA", "N/A"}
found = []
for c in text_cols:
    vals = train_df[c].astype(str).str.strip()
    if vals.isin(placeholders).any() or vals.eq("").any():
        found.append(c)
if found:
    print(f"Placeholder tokens found in {len(found)} text columns.")
else:
    print("No text columns with placeholder tokens.")

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

threshold = 0.05 * len(train_df)
high_card = [c for c in categorical_cols if train_df[c].nunique(dropna=False) > threshold]
for c in high_card:
    feature_cols.remove(c)
    categorical_cols.remove(c)
if high_card:
    print(f"Dropped high-cardinality categorical features: {len(high_card)}")

log_cols, plain_cols = [], []
for c in numeric_cols:
    s = train_df[c]
    if s.min() >= 0 and s.skew() > 1.0:
        log_cols.append(c)
    else:
        plain_cols.append(c)

transformers = []
if log_cols:
    transformers.append(("log", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", StandardScaler()),
    ]), log_cols))
if plain_cols:
    transformers.append(("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), plain_cols))
if categorical_cols:
    transformers.append(("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), categorical_cols))

preprocessor = ColumnTransformer(transformers, remainder="drop")
preprocessor.set_output(transform="pandas")
for _, trans, _ in transformers:
    if hasattr(trans, "set_output"):
        trans.set_output(transform="pandas")

X_train = train_df[feature_cols]
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, "train")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
print(f"Saved preprocessor with {len(feature_cols)} features.")
