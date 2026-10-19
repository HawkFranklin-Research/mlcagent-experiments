import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column", None)
task = protocol["task"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def key_like(s):
    if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s):
        return s.notna().all() and s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s) and s.notna().all() and s.nunique(dropna=True) == len(s):
        vals = np.sort(s.unique())
        return len(vals) > 1 and np.all(np.diff(vals) == 1)
    return False

block = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in block and not key_like(train[c])]

date_cols = [c for c in feature_cols if pd.api.types.is_datetime64_any_dtype(train[c])]
ref = pd.Timestamp("1970-01-01")
for c in date_cols:
    for suffix, func in [
        ("year", lambda x: x.dt.year),
        ("month", lambda x: x.dt.month),
        ("dow", lambda x: x.dt.dayofweek),
        ("days", lambda x: (x - ref).dt.days),
    ]:
        train[f"{c}_{suffix}"] = func(train[c])
        val[f"{c}_{suffix}"] = func(val[c])
        feature_cols.append(f"{c}_{suffix}")
    feature_cols.remove(c)

text_cols = [
    c for c in feature_cols
    if pd.api.types.is_object_dtype(train[c])
    or isinstance(train[c].dtype, pd.CategoricalDtype)
    or pd.api.types.is_string_dtype(train[c])
]
tokens = ["?", "-", "NA", "N/A"]
found = {}
for c in text_cols:
    s = train[c].astype(str).str.strip()
    n = int((s.isin(tokens) | s.eq("")).sum())
    if n:
        found[c] = n
    for df in (train, val):
        df[c] = df[c].astype(str).str.strip().replace(tokens + [""], np.nan)

numeric_features, categorical_features, dropped_high = [], [], []
n_train = len(train)
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        numeric_features.append(c)
    elif pd.api.types.is_datetime64_any_dtype(s):
        dropped_high.append(c)
    else:
        if s.nunique(dropna=True) > 0.05 * n_train:
            dropped_high.append(c)
        else:
            categorical_features.append(c)
feature_cols = numeric_features + categorical_features

excluded_derived = []
if len(targets) == 1 and pd.api.types.is_numeric_dtype(train[targets[0]]):
    y = train[targets[0]]
    for c in numeric_features:
        if train[c].dtype.kind in "if" and (train[c] == y).all():
            excluded_derived.append(c)
    if excluded_derived:
        feature_cols = [c for c in feature_cols if c not in excluded_derived]
        numeric_features = [c for c in numeric_features if c not in excluded_derived]

width = len(numeric_features) + sum(train[c].nunique(dropna=True) for c in categorical_features)
if width <= 300:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]).set_output(transform="pandas")
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]).set_output(transform="pandas")
else:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

transformers = []
if numeric_features:
    transformers.append(("num", num_pipe, numeric_features))
if categorical_features:
    transformers.append(("cat", cat_pipe, categorical_features))
preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
if width <= 300:
    preprocessor.set_output(transform="pandas")

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, "train")
X_val_trans = preprocessor.transform(X_val)

print(f"features={len(feature_cols)} numeric={len(numeric_features)} categorical={len(categorical_features)} dropped={len(dropped_high)}")
print("placeholders: " + (str(found) if found else "none"))
print("derived exclusions: " + (str(excluded_derived) if excluded_derived else "none"))
print(f"val_transformed shape={getattr(X_val_trans, 'shape', 'unknown')}")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
