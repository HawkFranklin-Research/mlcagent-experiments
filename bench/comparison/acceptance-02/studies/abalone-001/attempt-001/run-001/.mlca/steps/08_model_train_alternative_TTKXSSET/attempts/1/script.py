import json
import mlca_runtime
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol)}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
train = pd.read_parquet(inputs["train"])

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

feature_cols = []
n_rows = len(train)
for col in train.columns:
    if col in drop:
        continue
    s = train[col]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    nunique = s.nunique(dropna=True)
    if n_rows > 1 and nunique == n_rows:
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            continue
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if nunique > 0.05 * n_rows:
            continue
    feature_cols.append(col)

if not feature_cols:
    raise ValueError("No feature columns remain after exclusions")

X = train[feature_cols].copy()
y = train[targets].copy()

num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]

width = len(num_cols) + sum(X[c].nunique(dropna=True) for c in cat_cols)
transformers = []
if num_cols:
    transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
if cat_cols:
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False if width <= 300 else True)),
    ])
    transformers.append(("cat", cat_pipe, cat_cols))

pre = ColumnTransformer(transformers, remainder="drop")
model = RandomForestRegressor(n_estimators=100, random_state=42)
pipe = Pipeline([("pre", pre), ("model", model)])
if width <= 300:
    pre.set_output(transform="pandas")
    pipe.set_output(transform="pandas")

fitted = mlca_runtime.fit_record(pipe, X, partition="train", y=y)
mlca_runtime.save_model(fitted, "alternative_model.pkl")
print("features", len(feature_cols), "rows", len(X))
print("saved alternative_model.pkl")
