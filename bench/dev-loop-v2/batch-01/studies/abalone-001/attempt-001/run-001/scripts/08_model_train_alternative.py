import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_column = protocol["group_column"]
_ = protocol["task"]

train = pd.read_parquet(inputs["train"])

def is_key_like(s: pd.Series) -> bool:
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        vals = np.sort(s.dropna().unique())
        if len(vals) == len(s) and len(vals) > 1:
            return np.array_equal(vals, np.arange(vals[0], vals[0] + len(vals)))
    return False

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or (group_column is not None and c == group_column):
        continue
    if is_key_like(train[c]):
        continue
    feature_cols.append(c)

num_cols = [
    c for c in feature_cols
    if pd.api.types.is_numeric_dtype(train[c].dtype) or pd.api.types.is_bool_dtype(train[c].dtype)
]
cat_cols = [c for c in feature_cols if c not in num_cols]
max_cat = max(5, int(0.05 * len(train)))
low_cat_cols = [c for c in cat_cols if train[c].nunique(dropna=True) <= max_cat]
feature_cols = num_cols + low_cat_cols

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

transformers = []
if num_cols:
    transformers.append(("num", numeric_transformer, num_cols))
if low_cat_cols:
    transformers.append(("cat", categorical_transformer, low_cat_cols))

pre = ColumnTransformer(
    transformers,
    remainder="drop",
    verbose_feature_names_out=False,
)
pre.set_output(transform="pandas")

pipe = Pipeline([
    ("pre", pre),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
])
pipe.set_output(transform="pandas")

fitted = mlca_runtime.fit_record(
    pipe,
    train[feature_cols],
    "train",
    y=train[targets],
)
mlca_runtime.save_model(fitted, "alternative_model.pkl")

print(f"features={len(feature_cols)} numeric={len(num_cols)} categorical={len(low_cat_cols)}")
print(f"train_rows={len(train)} targets={targets}")
print("saved=alternative_model.pkl")
