import json
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlca_runtime.set_seed(42)

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {sorted(d.keys())}")
    return d[key]

inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = need(proto, "targets")
excluded = need(proto, "excluded_columns")
task = need(proto, "task")
_ = need(task, "primary_metric")
group_col = proto["group_column"] if "group_column" in proto else None

train = pd.read_parquet(inputs["train"])
missing = [c for c in targets if c not in train.columns]
if missing:
    raise KeyError(f"Missing target columns in training table: {missing}")

y = train[targets].copy()
if not all(pd.api.types.is_numeric_dtype(y[c]) for c in targets):
    raise TypeError("All target columns must be numeric")

def key_like(s):
    n = len(s)
    nu = int(s.nunique(dropna=True))
    if n <= 1 or nu <= 1 or nu != n:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s):
        vals = set(pd.unique(s.dropna()))
        return not vals.issubset({0, 1})
    return False

skip = set(targets) | set(excluded)
if group_col:
    skip.add(group_col)
feature_cols = [c for c in train.columns if c not in skip and not key_like(train[c])]
if not feature_cols:
    raise ValueError("No feature columns remain after excluding targets, exclusions and keys")

X = train[feature_cols].copy()
num_cols, cat_cols = [], []
for c in feature_cols:
    s = X[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        nu = int(s.nunique(dropna=True))
        if nu <= 5 or nu <= 0.05 * max(len(X), 1):
            cat_cols.append(c)

feature_cols = num_cols + cat_cols
if not feature_cols:
    raise ValueError("No usable feature columns after type sorting")
X = X[feature_cols].copy()

transformers = []
if num_cols:
    transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
if cat_cols:
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers.append(("cat", cat_pipe, cat_cols))

ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
width = len(num_cols)
for c in cat_cols:
    width += int(X[c].nunique(dropna=True)) + 1
if width <= 300:
    ct.set_output(transform="pandas")

pipe = Pipeline([
    ("preprocessor", ct),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
])

fitted = mlca_runtime.fit_record(pipe, X, partition="train", y=y)
mlca_runtime.save_model(fitted, "alternative_model.pkl")
print(f"train rows={len(train)}, features={len(feature_cols)}, targets={targets}")
