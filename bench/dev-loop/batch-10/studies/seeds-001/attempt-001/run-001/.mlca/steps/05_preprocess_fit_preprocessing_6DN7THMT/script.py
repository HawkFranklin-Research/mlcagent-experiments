import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
task = protocol["task"]

cols = [c for c in train.columns if c not in targets and c not in excluded]
if group in cols:
    cols.remove(group)

def is_identifier(c):
    s = train[c]
    if s.dtype == object:
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s):
        name = c.lower()
        return s.nunique(dropna=True) == len(s) and any(k in name for k in ("id", "key", "index", "row"))
    return False

features = [c for c in cols if not is_identifier(c)]
num_cols, cat_cols, date_cols = [], [], []
for c in features:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        cat_cols.append(c)

if date_cols:
    for c in date_cols:
        dt = pd.to_datetime(train[c], errors="coerce")
        for part in ("year", "month", "day", "dayofweek"):
            name = f"{c}_{part}"
            train[name] = getattr(dt.dt, part)
            num_cols.append(name)
    features = [c for c in features if c not in date_cols]

log_cols = []
for c in num_cols:
    s = train[c].dropna()
    if len(s) and (s >= 0).all() and s.skew() > 2:
        log_cols.append(c)

transformers = []
for c in num_cols:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if c in log_cols:
        steps.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    steps.append(("scaler", StandardScaler()))
    transformers.append((f"num_{c}", Pipeline(steps), [c]))

for c in cat_cols:
    if train[c].nunique(dropna=True) > 0.05 * len(train):
        continue
    transformers.append((
        f"cat_{c}",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]),
        [c],
    ))

pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
pre.set_output(transform="pandas")
fitted = mlca_runtime.fit_record(pre, train[features], "train")

print(f"features={len(features)} numeric={len(num_cols)} categorical={len(cat_cols)} log1p={len(log_cols)}")
print(f"date_columns={len(date_cols)}")
mlca_runtime.save_model(fitted, "preprocessor.pkl")
