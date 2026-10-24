import mlca_runtime
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

def need(d, key):
    if key not in d:
        raise KeyError(f"missing {key}; available top-level keys: {list(d)}")
    return d[key]

targets = need(protocol, "targets")
excluded = set(need(protocol, "excluded_columns"))
group = protocol.get("group_column")
train = pd.read_parquet(paths["train"])

def is_key_like(s):
    n = len(s)
    if n == 0:
        return False
    nu = s.nunique(dropna=True)
    if pd.api.types.is_integer_dtype(s):
        name = str(s.name).lower()
        return nu == n and any(t in name for t in ("id", "key", "code", "index"))
    if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s):
        return nu == n
    return False

candidate_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if is_key_like(s):
        continue
    candidate_cols.append(c)

num_cols, cat_cols = [], []
for c in candidate_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    elif s.nunique(dropna=True) <= max(5, 0.05 * len(train)):
        cat_cols.append(c)

feature_cols = num_cols + cat_cols

num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))

pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
pre.set_output(transform="pandas")

pipe = Pipeline([
    ("prep", pre),
    ("model", MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))),
])
pipe.set_output(transform="pandas")

X = train[feature_cols].copy()
y = train[targets].copy()
fitted = mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(fitted, "alternative_model.pkl")
print("train_rows", len(train), "features", len(feature_cols), "targets", len(targets))
