import json
import pickle
from pathlib import Path

import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(Path(inputs["_protocol"]).read_text())

for key in ("targets", "excluded_columns"):
    if key not in protocol:
        raise KeyError(f"Missing {key!r} in _protocol; available keys: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
forbidden = set(targets) | set(excluded)
if group is not None:
    forbidden.add(group)

train = pd.read_parquet(inputs["train"])

feature_cols = None
try:
    with open(inputs["05_preprocess_fit_preprocessing_GTAJ2HZB/preprocessor.pkl"], "rb") as f:
        preprocessor = pickle.load(f)
    feature_cols = getattr(preprocessor, "feature_names_in_", None)
except Exception:
    feature_cols = None

if feature_cols is None:
    manifest = json.loads(Path(inputs["05_preprocess_fit_preprocessing_GTAJ2HZB/manifest.json"]).read_text())

    def find_feature_list(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                    if any(tok in k.lower() for tok in ("feature", "column", "input")):
                        return v
            for v in obj.values():
                found = find_feature_list(v)
                if found:
                    return found
        elif isinstance(obj, list):
            for v in obj:
                found = find_feature_list(v)
                if found:
                    return found
        return None

    feature_cols = find_feature_list(manifest)

if feature_cols is None:
    feature_cols = []
    for c in train.columns:
        if c in forbidden:
            continue
        s = train[c]
        all_distinct = s.nunique(dropna=False) == len(train)
        key_like = all_distinct and (
            pd.api.types.is_integer_dtype(s)
            or pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
        )
        if not key_like:
            feature_cols.append(c)

feature_cols = [c for c in feature_cols if c not in forbidden]
missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise KeyError(f"Declared feature columns missing from train: {missing}")
if not feature_cols:
    raise ValueError("No feature columns remain after applying the declared contract.")

X_train = train.loc[:, feature_cols].copy()
y_train = train.loc[:, targets].astype(float).copy()

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_train[c])]
cat_cols = [c for c in feature_cols if c not in numeric_cols]

width_est = len(numeric_cols) + sum(X_train[c].nunique(dropna=False) for c in cat_cols)
if width_est > 300:
    max_card = max(5, int(0.05 * len(X_train)))
    cat_cols = [c for c in cat_cols if X_train[c].nunique(dropna=False) <= max_card]
    width_est = len(numeric_cols) + sum(X_train[c].nunique(dropna=False) for c in cat_cols)

use_pandas = width_est <= 300
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not use_pandas)),
])

transformers = []
if numeric_cols:
    transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
if not transformers:
    raise ValueError("No usable feature transformers were built.")

ct = ColumnTransformer(transformers, remainder="drop")
if use_pandas:
    ct.set_output(transform="pandas")

model = Pipeline([
    ("prep", ct),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
])

mlca_runtime.fit_record(model, X_train, "train", y=y_train)
mlca_runtime.save_model(model, "alternative_model.pkl")

print(f"train rows={len(train)}, feature columns={len(feature_cols)}")
print(f"targets={targets}")
print("saved alternative_model.pkl")
