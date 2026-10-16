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
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

top_keys = list(protocol.keys())
if "targets" not in protocol:
    raise KeyError(f"Missing 'targets'. Available top-level keys: {top_keys}")
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column", None)

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = [c for c in train.columns if c not in drop]

key_like = []
for c in feature_cols:
    s = train[c]
    n = s.nunique(dropna=True)
    if n <= 1:
        continue
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if n == len(s):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        if n == len(s):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

excluded_features = {}
for t in targets:
    if t not in train.columns:
        continue
    y = train[t]
    for c in list(feature_cols):
        if c in excluded_features:
            continue
        if train[c].equals(y):
            excluded_features[c] = f"exact equality with {t}"
    nonneg = [c for c in feature_cols if c not in excluded_features
              and pd.api.types.is_numeric_dtype(train[c])
              and (train[c].dropna() >= 0).all()]
    for i in range(len(nonneg)):
        for j in range(i + 1, len(nonneg)):
            if (train[nonneg[i]] + train[nonneg[j]]).equals(y):
                excluded_features[nonneg[j]] = f"sum with {nonneg[i]} equals {t}"
    for i in range(len(nonneg)):
        for j in range(i + 1, len(nonneg)):
            for k in range(j + 1, len(nonneg)):
                if (train[nonneg[i]] + train[nonneg[j]] + train[nonneg[k]]).equals(y):
                    excluded_features[nonneg[k]] = f"sum equals {t}"
for c, r in excluded_features.items():
    print(f"Excluded feature {c}: {r}")
feature_cols = [c for c in feature_cols if c not in excluded_features]

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.astype(str).str.strip().isin(placeholders).any():
            ph_cols.append(c)
if ph_cols:
    print(f"Placeholder tokens found in {len(ph_cols)} columns")

numeric_cols, cat_cols = [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    else:
        if s.nunique(dropna=True) > 0.05 * len(train):
            continue
        cat_cols.append(c)

est_width = len(numeric_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
if est_width > 300:
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    preprocessor = ColumnTransformer([("num", num_pipe, numeric_cols),
                                      ("cat", cat_pipe, cat_cols)], remainder="drop")
else:
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer([("num", num_pipe, numeric_cols),
                                      ("cat", cat_pipe, cat_cols)], remainder="drop")
    preprocessor.set_output(transform="pandas")

X_train = train[feature_cols]
mlca_runtime.fit_record(preprocessor, X_train, "train")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
print(f"Features: {len(feature_cols)} ({len(numeric_cols)} num, {len(cat_cols)} cat), est_width={est_width}")
