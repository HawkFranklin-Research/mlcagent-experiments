import json
import mlca_runtime
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing {key!r}; available keys: {list(d.keys())}")
    return d[key]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns") or []
group = need(protocol, "group_column")
task = need(protocol, "task")
_ = need(task, "primary_metric")

train = pd.read_parquet(inputs["train"])
for c in targets:
    if c not in train.columns:
        raise KeyError(f"Target {c!r} not in training table; columns: {list(train.columns)}")

def key_like(s):
    if s.notna().sum() != len(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == len(s)
    return False

feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group and not key_like(train[c])
]
if not feature_cols:
    raise ValueError("No feature columns remain after applying the contract")

X = train[feature_cols].copy()
y = train[targets].copy()

num_cols = [
    c for c in feature_cols
    if pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c])
]
cat_cols = [c for c in feature_cols if c not in num_cols]
width = len(num_cols) + sum(int(train[c].nunique(dropna=True)) for c in cat_cols)
wide = width > 300

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=wide)),
])
transformers = []
if num_cols:
    transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))

prep = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
if not wide:
    prep.set_output(transform="pandas")

model = Pipeline([
    ("prep", prep),
    ("model", MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))),
])
if not wide:
    model.set_output(transform="pandas")

fitted = mlca_runtime.fit_record(model, X, "train", y=y)
if wide:
    _ = fitted.named_steps["prep"].get_feature_names_out()
mlca_runtime.save_model(fitted, "alternative_model.pkl")
print(f"train rows={len(X)}, features={len(feature_cols)}, targets={len(targets)}")
print(f"encoded width estimate={width}, sparse={wide}")
