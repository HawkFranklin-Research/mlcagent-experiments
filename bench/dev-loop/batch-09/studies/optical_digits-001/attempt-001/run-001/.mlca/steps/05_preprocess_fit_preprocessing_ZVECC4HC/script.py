import mlca_runtime
import json
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_integer_dtype
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "group_column"):
    if key not in protocol:
        raise KeyError(f"Missing {key!r}; available top-level keys: {sorted(protocol)}")

targets = set(protocol["targets"] or [])
excluded = set(protocol["excluded_columns"] or [])
group_col = protocol["group_column"]

train = pd.read_parquet(inputs["train"])
feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = train[c]
    if s.dtype == object:
        if s.nunique(dropna=True) == len(s):
            continue
    elif is_integer_dtype(s.dtype):
        vals = s.dropna().unique()
        if s.nunique(dropna=True) == len(s) and not set(vals).issubset({0, 1}):
            continue
    feature_cols.append(c)

placeholder_cols = []
for c in feature_cols:
    if train[c].dtype == object:
        vals = train[c].astype("string").str.strip()
        if vals.isin(["?", "-", "NA", "N/A", ""]).any():
            placeholder_cols.append(c)

num_cols = [c for c in feature_cols if is_numeric_dtype(train[c])]
cat_all = [
    c for c in feature_cols
    if c not in num_cols and (train[c].dtype == object or isinstance(train[c].dtype, pd.CategoricalDtype))
]
cat_cols = [c for c in cat_all if train[c].nunique(dropna=True) <= 0.05 * len(train)]

log_cols, plain_num = [], []
for c in num_cols:
    s = train[c]
    if s.min() >= 0 and s.skew() > 1:
        log_cols.append(c)
    else:
        plain_num.append(c)

transformers = []
if log_cols:
    transformers.append((
        "num_log",
        Pipeline([
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]).set_output(transform="pandas"),
        log_cols,
    ))
if plain_num:
    transformers.append((
        "num",
        Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]).set_output(transform="pandas"),
        plain_num,
    ))
if cat_cols:
    transformers.append((
        "cat",
        Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]).set_output(transform="pandas"),
        cat_cols,
    ))

preprocessor = ColumnTransformer(transformers, remainder="drop").set_output(transform="pandas")
preprocessor = mlca_runtime.fit_record(preprocessor, train[feature_cols], "train")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

print(f"features: {len(feature_cols)}")
print(f"numeric/log1p/categorical: {len(plain_num)}/{len(log_cols)}/{len(cat_cols)}")
print(f"placeholder columns: {len(placeholder_cols)}")
