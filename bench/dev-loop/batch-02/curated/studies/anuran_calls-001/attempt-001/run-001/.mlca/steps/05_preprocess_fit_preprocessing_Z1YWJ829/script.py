import json
import mlca_runtime
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])

for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing {key}. Available keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]

features = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]

# key-like columns: text codes or integer sequences with all distinct values
key_like = []
for c in list(features):
    s = train[c]
    if (s.dtype == object or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)):
        if s.nunique(dropna=False) == len(s):
            key_like.append(c)
            features.remove(c)
if key_like:
    print(f"excluded key-like columns: {key_like}")

# placeholder tokens
placeholders = ["?", "-", "NA", "N/A", ""]
placeholder_cols = []
for c in features:
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        stripped = s.dropna().astype(str).str.strip()
        hits = [p for p in placeholders if (stripped == p).any()]
        if hits:
            placeholder_cols.append(c)
            train[c] = s.astype(str).str.strip().replace(placeholders, np.nan)
if placeholder_cols:
    print(f"placeholders found in {len(placeholder_cols)} columns: {placeholder_cols}")

# exact derivations of a target
excluded_deriv = []
for t in targets:
    for c in list(features):
        if c == t:
            continue
        if (train[c] == train[t]).all():
            excluded_deriv.append(c)
            features.remove(c)
if excluded_deriv:
    print(f"excluded exact target derivations: {excluded_deriv}")

# correlation warnings for numeric targets only
warned = []
for t in targets:
    if pd.api.types.is_numeric_dtype(train[t]):
        for c in features:
            if pd.api.types.is_numeric_dtype(train[c]):
                corr = train[[c, t]].corr().iloc[0, 1]
                if abs(corr) >= 0.98:
                    warned.append((c, t, round(corr, 4)))
if warned:
    print(f"high-correlation warnings: {warned}")

# split feature types
num_cols, cat_cols, drop_cols = [], [], []
for c in features:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    elif s.dtype == object or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        n = s.nunique(dropna=True)
        if n > 0.05 * len(train) and n > 20:
            drop_cols.append(c)
        else:
            cat_cols.append(c)
    else:
        num_cols.append(c)

if drop_cols:
    print(f"dropped high-cardinality columns: {drop_cols}")
    features = [c for c in features if c not in drop_cols]

n_out = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
sparse = n_out > 300

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)),
])

transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))

pre = ColumnTransformer(transformers, remainder="drop")
if not sparse:
    pre.set_output(transform="pandas")

fitted = mlca_runtime.fit_record(pre, train[features], "train")
mlca_runtime.save_model(fitted, "preprocessor.pkl")
print(f"preprocessor fitted on {len(features)} features, output width {n_out}")
