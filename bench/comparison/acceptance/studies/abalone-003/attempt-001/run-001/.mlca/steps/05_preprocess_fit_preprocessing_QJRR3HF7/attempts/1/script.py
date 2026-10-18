import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    task = protocol["task"]
except KeyError as e:
    raise KeyError(f"Missing key {e} in _protocol; available keys: {list(protocol.keys())}")
group = protocol.get("group_column")

protected = set(targets) | set(excluded) | ({group} if group else set())
candidate = [c for c in train.columns if c not in protected]
key_like = []
for c in candidate:
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)) and s.nunique(dropna=False) == len(train):
        key_like.append(c)
feature_cols = [c for c in candidate if c not in key_like]

excluded_deriv = set()
for target in targets:
    if target not in train:
        continue
    y = train[target]
    for c in feature_cols:
        if train[c].equals(y):
            excluded_deriv.add(c)
    nf = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and (train[c].dropna() >= 0).all()]
    for i in range(len(nf)):
        for j in range(i + 1, len(nf)):
            if np.allclose((train[nf[i]] + train[nf[j]]).to_numpy(), y.to_numpy(), equal_nan=True):
                excluded_deriv.update([nf[i], nf[j]])
            for k in range(j + 1, len(nf)):
                if np.allclose((train[nf[i]] + train[nf[j]] + train[nf[k]]).to_numpy(), y.to_numpy(), equal_nan=True):
                    excluded_deriv.update([nf[i], nf[j], nf[k]])
if excluded_deriv:
    print(f"Excluded derivations of target: {sorted(excluded_deriv)}")
feature_cols = [c for c in feature_cols if c not in excluded_deriv]

warn = []
for target in targets:
    if pd.api.types.is_numeric_dtype(train[target]) and not pd.api.types.is_bool_dtype(train[target]):
        for c in feature_cols:
            if pd.api.types.is_numeric_dtype(train[c]):
                corr = train[[c, target]].dropna().corr().iloc[0, 1]
                if abs(corr) >= 0.98:
                    warn.append(c)
if warn:
    print(f"High correlation with target (kept): {sorted(set(warn))}")

placeholders = {"?", "-", "NA", "N/A", ""}
text_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
ph_cols, found_tokens = [], set()
for c in text_cols:
    s = train[c].astype("string")
    stripped = s.str.strip()
    for tok in placeholders:
        if (stripped == tok).any():
            found_tokens.add(tok)
    if (stripped.isin(placeholders) | (s != stripped)).any():
        ph_cols.append(c)
if ph_cols:
    print(f"Placeholders {sorted(found_tokens)} in {len(ph_cols)} columns: {ph_cols}")
else:
    print("No placeholder tokens found")

numeric_cols, cat_cols, drop_cols = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_bool_dtype(s) or (pd.api.types.is_numeric_dtype(s) and set(s.dropna().unique()).issubset({0, 1})):
        numeric_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    elif pd.api.types.is_datetime64_any_dtype(s):
        numeric_cols.append(c)
    elif s.nunique(dropna=False) > 0.05 * len(train):
        drop_cols.append(c)
    else:
        cat_cols.append(c)
feature_cols = [c for c in feature_cols if c not in drop_cols]

num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
transformers = []
if numeric_cols:
    transformers.append(("num", num_pipe, numeric_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
preprocessor = ColumnTransformer(transformers, remainder="drop")
preprocessor.set_output(transform="pandas")
num_pipe.set_output(transform="pandas")
cat_pipe.set_output(transform="pandas")

X_train = train[feature_cols].copy()
mlca_runtime.fit_record(preprocessor, X_train, partition="train")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
print(f"Saved preprocessor with {len(feature_cols)} features")
