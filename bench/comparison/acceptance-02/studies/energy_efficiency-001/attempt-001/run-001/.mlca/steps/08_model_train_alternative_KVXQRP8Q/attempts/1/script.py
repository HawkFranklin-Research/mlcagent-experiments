import json
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol)}")

targets = protocol["targets"]
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
train = pd.read_parquet(inp["train"])

def key_like(s):
    n = len(s)
    if n == 0 or s.isna().all():
        return False
    if s.nunique(dropna=True) != n:
        return False
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        return True
    if pd.api.types.is_integer_dtype(s.dtype):
        return True
    return False

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded:
        continue
    if group is not None and c == group:
        continue
    if key_like(train[c]):
        continue
    feature_cols.append(c)

num_cols, cat_cols = [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        continue
    if pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype):
        num_cols.append(c)
    else:
        n = len(s)
        if s.nunique(dropna=True) <= max(5, 0.05 * n):
            cat_cols.append(c)

width_est = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
transformers = []
if num_cols:
    transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
if cat_cols:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=(width_est > 300))
    transformers.append((
        "cat",
        Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]),
        cat_cols,
    ))
if not transformers:
    raise ValueError("No usable feature columns after applying the feature contract")

pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
if width_est <= 300:
    pre.set_output(transform="pandas")

model = Pipeline([
    ("pre", pre),
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
])

X = train[feature_cols].copy()
y = train[targets].copy()
mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(model, "alternative_model.pkl")

print("train rows:", len(train))
print("feature columns:", len(feature_cols))
print("saved alternative_model.pkl")
