import mlca_runtime
import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ["targets", "excluded_columns"]:
    if key not in protocol:
        raise KeyError(f"Missing {key}; available top-level keys: {list(protocol)}")

targets = protocol["targets"]
target = targets[0]
excluded_raw = protocol["excluded_columns"]
excluded = set(excluded_raw.get("examples", []) if isinstance(excluded_raw, dict) else excluded_raw)
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

feature_columns = [c for c in train.columns if c not in set(targets) and c not in excluded and c != group]
key_like = []
for c in list(feature_columns):
    s = train[c]
    textish = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)
    intish = pd.api.types.is_integer_dtype(s)
    if (textish or intish) and s.nunique(dropna=False) == len(train):
        feature_columns.remove(c)
        key_like.append(c)

placeholders = ["?", "-", "NA", "N/A", ""]
text_cols = [c for c in feature_columns if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
ph_cols = []
for c in text_cols:
    s = train[c].astype("string").str.strip()
    if s.isin(placeholders).any():
        ph_cols.append(c)
if ph_cols:
    for df in (train, val):
        for c in ph_cols:
            s = df[c].astype("string").str.strip()
            s = s.mask(s.isin(placeholders), np.nan)
            df[c] = s.astype(object)
    print(f"Placeholders found in {len(ph_cols)} columns")
else:
    print("Placeholders found: none")

hc_cols = []
for c in list(feature_columns):
    if c in text_cols and train[c].nunique(dropna=False) > 0.05 * len(train):
        feature_columns.remove(c)
        hc_cols.append(c)
if hc_cols:
    print(f"Dropped high-cardinality text columns: {len(hc_cols)}")

datetime_cols = []
for c in list(feature_columns):
    if pd.api.types.is_datetime64_any_dtype(train[c]):
        for df in (train, val):
            df[c + "_year"] = df[c].dt.year
            df[c + "_month"] = df[c].dt.month
            df[c + "_day"] = df[c].dt.day
        feature_columns.remove(c)
        feature_columns += [c + "_year", c + "_month", c + "_day"]
        datetime_cols.append(c)
if datetime_cols:
    print(f"Expanded datetime columns: {len(datetime_cols)}")

y = train[target]
derived = set()
for c in list(feature_columns):
    s = train[c]
    if s.notna().all() and (s == y).all():
        derived.add(c)

uniq_y = set(y.dropna().unique().tolist())
if uniq_y <= {0, 1, 0.0, 1.0}:
    bin_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(train[c]) and train[c].dropna().isin([0, 1]).all()]
    if len(bin_cols) <= 40:
        for r in (2, 3):
            for combo in combinations(bin_cols, r):
                ssum = train[list(combo)].sum(axis=1, skipna=False)
                if (ssum == y).all():
                    derived.update(combo)

for c in list(derived):
    if c in feature_columns:
        feature_columns.remove(c)
if derived:
    print(f"Excluded target-derived features: {len(derived)}")

if pd.api.types.is_numeric_dtype(y):
    corr_warn = []
    yv = y.astype(float)
    for c in feature_columns:
        if pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_datetime64_any_dtype(train[c]):
            s = train[c]
            mask = s.notna() & y.notna()
            if mask.sum() > 1:
                corr = np.corrcoef(s[mask].astype(float), yv[mask])[0, 1]
                if abs(corr) >= 0.98:
                    corr_warn.append(c)
    if corr_warn:
        print(f"High correlation >=0.98 kept: {corr_warn[:5]}")

numeric_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_datetime64_any_dtype(train[c])]
categorical_cols = [c for c in feature_columns if c not in numeric_cols]

width = len(numeric_cols) + sum(train[c].nunique(dropna=False) + 1 for c in categorical_cols)
use_pandas = width <= 300

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not use_pandas)),
])
if use_pandas:
    numeric_pipe.set_output(transform="pandas")
    categorical_pipe.set_output(transform="pandas")

transformers = []
if numeric_cols:
    transformers.append(("num", numeric_pipe, numeric_cols))
if categorical_cols:
    transformers.append(("cat", categorical_pipe, categorical_cols))
if not transformers:
    raise ValueError("No feature columns remain after screening")

preprocessor = ColumnTransformer(transformers, remainder="drop")
if use_pandas:
    preprocessor.set_output(transform="pandas")

X_train = train[feature_columns].copy()
X_val = val[feature_columns].copy()
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, "train", y)
X_val_t = preprocessor.transform(X_val)
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
mlca_runtime.record_metric("preprocessor_output_features", float(X_val_t.shape[1]), "val")
print(f"Features: {len(feature_columns)}; key-like dropped: {len(key_like)}; output shape: {X_val_t.shape}; pandas: {use_pandas}")
