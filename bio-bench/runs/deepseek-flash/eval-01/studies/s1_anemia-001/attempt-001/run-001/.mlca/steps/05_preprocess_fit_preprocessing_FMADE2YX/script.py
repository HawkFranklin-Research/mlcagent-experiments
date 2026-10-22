import mlca_runtime
import pandas as pd
import numpy as np
import json
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
if "group_column" not in protocol:
    raise KeyError(f"Missing group_column. Available keys: {list(protocol.keys())}")
group_col = protocol["group_column"]
_ = protocol["task"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
y = train[targets[0]]

# candidate features
cand = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
feature_cols = []
for c in cand:
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)) and s.nunique(dropna=True) == len(train):
        continue  # key-like text code
    feature_cols.append(c)

# screening: exact derivations and correlation warnings
excluded_features, reasons = [], {}
for c in feature_cols:
    if train[c].equals(y):
        excluded_features.append(c)
        reasons[c] = "exact equality with target"
numeric_cand = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and c not in excluded_features]
if pd.api.types.is_numeric_dtype(y):
    for r in (1, 2, 3):
        for combo in combinations(numeric_cand, r):
            if all((train[c] >= 0).all() for c in combo) and train[list(combo)].sum(axis=1).equals(y):
                for c in combo:
                    if c not in excluded_features:
                        excluded_features.append(c)
                        reasons[c] = f"row sum {combo} equals target"
    warn = [c for c in feature_cols if c not in excluded_features and pd.api.types.is_numeric_dtype(train[c]) and abs(train[c].corr(y)) >= 0.98]
    if warn:
        print(f"Correlation warning (>=0.98): {warn}")
if excluded_features:
    print(f"Excluded features: {excluded_features}")
feature_cols = [c for c in feature_cols if c not in excluded_features]

# classify
numeric_cols, cat_cols, drop_cols = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        drop_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    elif s.nunique(dropna=True) > 0.05 * len(train):
        drop_cols.append(c)
    else:
        cat_cols.append(c)
if drop_cols:
    print(f"Dropped columns: {drop_cols}")
feature_cols = numeric_cols + cat_cols

# placeholder detection
tokens = ["?", "-", "NA", "N/A", ""]
found = {}
for c in cat_cols:
    for v in train[c].dropna().unique():
        if isinstance(v, str) and v.strip() in tokens:
            found.setdefault(v.strip(), set()).add(c)
if found:
    summary = ", ".join(f"'{k}': {len(v)} cols" for k, v in found.items())
    print(f"Placeholders found: {summary}")

# encoded width
cat_width = sum(train[c].nunique(dropna=True) for c in cat_cols)
use_pandas = (len(numeric_cols) + cat_width) <= 300

cat_steps = [
    ("fill_sentinel", SimpleImputer(strategy="constant", fill_value="__MISSING__", missing_values=np.nan)),
    ("strip", FunctionTransformer(np.char.strip)),
]
for i, tok in enumerate(tokens):
    cat_steps.append((f"replace_{i}", SimpleImputer(missing_values=tok, strategy="constant", fill_value="__MISSING__")))
cat_steps += [
    ("sentinel_to_nan", SimpleImputer(missing_values="__MISSING__", strategy="constant", fill_value=np.nan)),
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not use_pandas)),
]
cat_pipe = Pipeline(cat_steps)
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

transformers = []
if numeric_cols:
    transformers.append(("num", num_pipe, numeric_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
preprocessor = ColumnTransformer(transformers, remainder="drop")
if use_pandas:
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")
    preprocessor.set_output(transform="pandas")

X_train = train[feature_cols]
X_val = val[feature_cols]
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, partition="train", y=y)
X_val_trans = preprocessor.transform(X_val)
X_train_trans = preprocessor.transform(X_train)
print(f"Transformed train shape: {X_train_trans.shape}, val shape: {X_val_trans.shape}")

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
