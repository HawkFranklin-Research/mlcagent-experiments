import mlca_runtime
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


def need(d, key):
    if key not in d:
        raise KeyError(f"Missing {key}; available: {list(d.keys())}")
    return d[key]


mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = list(need(protocol, "targets"))
excluded = list(need(protocol, "excluded_columns"))
group = need(protocol, "group_column")
task = need(protocol, "task")

train = pd.read_parquet(inputs["train"])
candidate = [c for c in train.columns if c not in targets and c not in excluded and c != group]

feature_cols = []
for c in candidate:
    s = train[c]
    is_text = s.dtype == object or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype)
    if is_text and len(s) > 1 and s.nunique(dropna=False) >= len(s):
        continue
    if pd.api.types.is_integer_dtype(s.dtype) and len(s) > 1 and s.nunique(dropna=False) >= len(s):
        continue
    feature_cols.append(c)

n = len(train)
feature_cols = [
    c for c in feature_cols
    if not (
        (train[c].dtype == object or pd.api.types.is_string_dtype(train[c].dtype) or isinstance(train[c].dtype, pd.CategoricalDtype))
        and train[c].nunique(dropna=False) > max(5, int(0.05 * n))
    )
]

num_cols, cat_cols = [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype):
        num_cols.append(c)
    else:
        cat_cols.append(c)

cat_width = sum(train[c].nunique(dropna=False) + 1 for c in cat_cols)
use_pandas = (len(num_cols) + cat_width) <= 300

transformers = []
if num_cols:
    transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
if cat_cols:
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not use_pandas)),
    ])
    transformers.append(("cat", cat_pipe, cat_cols))

prep = ColumnTransformer(transformers, remainder="drop")
model = Pipeline([
    ("prep", prep),
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
])
if use_pandas:
    prep.set_output(transform="pandas")
    model.set_output(transform="pandas")

X = train[feature_cols].copy()
y = train[targets].copy()
mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(model, "alternative_model.pkl")
print(f"features={len(feature_cols)} numeric={len(num_cols)} categorical={len(cat_cols)}")
print(f"train_rows={len(train)} targets={targets}")
