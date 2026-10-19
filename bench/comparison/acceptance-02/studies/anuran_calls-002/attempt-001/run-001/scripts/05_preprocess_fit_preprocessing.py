import json
from itertools import combinations

import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing {key}. Available top-level keys: {sorted(d.keys())}")
    return d[key]

targets = list(need(protocol, "targets"))
excluded = list(need(protocol, "excluded_columns"))
group = need(protocol, "group_column")

drop_base = set(targets) | set(excluded) | {group}
features = [c for c in train.columns if c not in drop_base]

# Screen for confirmed target derivations before fitting.
excluded_deriv = set()
reasons = {}
for t in targets:
    if t not in train.columns:
        continue
    for c in list(features):
        if train[c].equals(train[t]):
            excluded_deriv.add(c)
            reasons[c] = f"equals {t}"
    if pd.api.types.is_numeric_dtype(train[t]) and train[t].notna().all():
        num_cols = [
            c for c in features
            if pd.api.types.is_numeric_dtype(train[c])
            and train[c].notna().all()
            and (train[c] >= 0).all()
        ]
        for r in (1, 2, 3):
            for combo in combinations(num_cols, r):
                s = train[list(combo)].sum(axis=1)
                if s.notna().all() and s.eq(train[t]).all():
                    for c in combo:
                        excluded_deriv.add(c)
                        reasons[c] = f"sum to {t}"
features = [c for c in features if c not in excluded_deriv]

# Remove key-like unique text/integer columns, except continuous measurements.
for c in list(features):
    s = train[c]
    is_text = (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )
    if len(train) > 1 and s.nunique(dropna=True) == len(train):
        if pd.api.types.is_integer_dtype(s) or is_text:
            features.remove(c)
            reasons[c] = "key-like unique column"

# Drop high-cardinality text/code columns (never one-hot).
threshold = max(10, int(0.05 * len(train)))
for c in list(features):
    s = train[c]
    if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s):
        if s.nunique(dropna=True) > threshold:
            features.remove(c)
            reasons[c] = "high-cardinality text/code"

# Drop text columns carrying placeholder tokens; no custom pickled cleaner is used.
placeholders = {"?", "-", "NA", "N/A"}
ph_cols = []
for c in list(features):
    s = train[c]
    if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s):
        stripped = s.astype(str).str.strip()
        has_ph = stripped.isin(placeholders).any() or stripped.eq("").any()
        if has_ph:
            ph_cols.append(c)
            features.remove(c)
            reasons[c] = "placeholder text"

X = train[features].copy()
date_cols = [c for c in features if pd.api.types.is_datetime64_any_dtype(X[c])]
for c in date_cols:
    dt = pd.to_datetime(X[c], errors="coerce")
    X[f"{c}_year"] = dt.dt.year
    X[f"{c}_month"] = dt.dt.month
    X[f"{c}_dayofweek"] = dt.dt.dayofweek
    X = X.drop(columns=[c])
features = list(X.columns)

if not features:
    raise ValueError("No feature columns remain after exclusions")

numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
categorical_features = [c for c in features if c not in numeric_features]

expected_width = len(numeric_features)
for c in categorical_features:
    expected_width += max(1, X[c].nunique(dropna=True))

use_pandas = expected_width <= 300
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=not use_pandas,
    )),
])

if use_pandas:
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")

transformers = []
if numeric_features:
    transformers.append(("num", num_pipe, numeric_features))
if categorical_features:
    transformers.append(("cat", cat_pipe, categorical_features))

preprocessor = ColumnTransformer(
    transformers,
    remainder="drop",
    verbose_feature_names_out=False,
)
if use_pandas:
    preprocessor.set_output(transform="pandas")

preprocessor = mlca_runtime.fit_record(preprocessor, X, "train")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

mlca_runtime.record_metric("n_features", float(len(features)), "train")
mlca_runtime.record_metric("n_derivation_exclusions", float(len(excluded_deriv)), "train")
print(f"Derivation exclusions: {len(excluded_deriv)}" + (": " + ", ".join(f"{c}={reasons[c]}" for c in sorted(excluded_deriv)) if excluded_deriv else ""))
print(f"Other exclusions: {len(reasons) - len(excluded_deriv)}")
print(f"Placeholders found: {len(ph_cols)} columns")
print(f"Saved preprocessor.pkl with {len(features)} features")
