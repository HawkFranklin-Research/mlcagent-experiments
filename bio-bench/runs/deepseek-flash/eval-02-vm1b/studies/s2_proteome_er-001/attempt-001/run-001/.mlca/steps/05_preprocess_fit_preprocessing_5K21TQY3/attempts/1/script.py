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
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
task = protocol.get("task", {})

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

feature_columns = [c for c in train.columns if c not in targets and c not in excluded and c != group]
key_like = []
for c in list(feature_columns):
    s = train[c]
    textish = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
    intish = pd.api.types.is_integer_dtype(s)
    if (textish or intish) and s.nunique(dropna=False) == len(train):
        feature_columns.remove(c)
        key_like.append(c)

text_cols = [c for c in feature_columns if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
placeholders = ["?", "-", "NA", "N/A", ""]
ph_cols = []
for c in text_cols:
    s = train[c].astype("string").str.strip()
    if s.isin(placeholders).any():
        ph_cols.append(c)
if ph_cols:
    for df in (train, val):
        for c in ph_cols:
            s = df[c].astype("string").str.strip()
            df[c] = s.mask(s.isin(placeholders), pd.NA)
    print(f"Placeholders found in {len(ph_cols)} columns")
else:
    print("Placeholders found: none")

hc = [c for c in text_cols if c in feature_columns and train[c].nunique(dropna=False) > 0.05 * len(train)]
for c in hc:
    feature_columns.remove(c)
if hc:
    print(f"Dropped high-cardinality text columns: {len(hc)}")

for c in list(feature_columns):
    if pd.api.types.is_datetime64_any_dtype(train[c]):
        for df in (train, val):
            df[c + "_year"] = df[c].dt.year
            df[c + "_month"] = df[c].dt.month
            df[c + "_day"] = df[c].dt.day
        feature_columns.remove(c)
        feature_columns += [c + "_year", c + "_month", c + "_day"]

y = train[target]
derived = []
for c in feature_columns:
    s = train[c]
    mask = s.notna() & y.notna()
    if mask.all() and (s == y).all():
        derived.append(c)
        continue
    if pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(y) and (y.dropna() >= 0).all():
        nonneg = [x for x in feature_columns if pd.api.types.is_numeric_dtype(train[x]) and (train[x].dropna() >= 0).all()]
    else:
        nonneg = []
    if 2 <= len(nonneg) <= 20:
        for r in (2, 3):
            for combo in combinations(nonneg, r):
                if c in combo:
                    ssum = train[list(combo)].sum(axis=1, skipna=False)
                    mask = ssum.notna() & y.notna()
                    if mask.all() and (ssum == y).all():
                        derived.extend([x for x in combo if x not in derived])
                        break
            if c in derived:
                break
for c in set(derived):
    if c in feature_columns:
        feature_columns.remove(c)
if derived:
    print(f"Excluded target-derived features: {sorted(set(derived))[:5]}")

if pd.api.types.is_numeric_dtype(y):
    corr_warn = []
    for c in feature_columns:
        if pd.api.types.is_numeric_dtype(train[c]):
            mask = train[c].notna() & y.notna()
            if mask.sum() > 1:
                corr = np.corrcoef(train.loc[mask, c], y[mask])[0, 1]
                if abs(corr) >= 0.98:
                    corr_warn.append(c)
    if corr_warn:
        print(f"High correlation >=0.98 kept: {corr_warn[:5]}")

numeric_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(train[c])]
categorical_cols = [c for c in feature_columns if c not in numeric_cols]

n_est = len(numeric_cols)
for c in categorical_cols:
    n_est += train[c].nunique(dropna=False) + 1
use_pandas = n_est <= 300

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
mlca_runtime.record_metric("preprocessor_output_features", int(X_val_t.shape[1]), "train")
print(f"Features: {len(feature_columns)}; key-like dropped: {len(key_like)}; output shape: {X_val_t.shape}; pandas: {use_pandas}")
