import mlca_runtime
import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

seed = 42
mlca_runtime.set_seed(seed)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
task = protocol["task"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
n = len(train)

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    if s.isna().all():
        continue
    nu = s.nunique(dropna=True)
    if nu == n and n > 10 and (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        continue
    if nu == n and n > 10 and pd.api.types.is_integer_dtype(s):
        continue
    feature_cols.append(c)

num_cols, cat_cols, date_cols = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    else:
        nu = s.nunique(dropna=True)
        if nu <= max(20, 0.05 * n):
            cat_cols.append(c)

for c in date_cols:
    d = pd.to_datetime(train[c], errors="coerce")
    train[f"{c}_year"] = d.dt.year
    train[f"{c}_month"] = d.dt.month
    train[f"{c}_dow"] = d.dt.dayofweek
    num_cols.extend([f"{c}_year", f"{c}_month", f"{c}_dow"])

feature_cols = [c for c in feature_cols if c not in date_cols]
feature_cols += [f"{c}_{p}" for c in date_cols for p in ("year", "month", "dow")]

y_train = train[targets[0]]
X_train = train[feature_cols].copy()

def make_pre():
    ts = []
    if num_cols:
        ts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols))
    if cat_cols:
        ts.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols))
    return ColumnTransformer(ts, remainder="drop")

pre = make_pre()
width = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
if width <= 300:
    pre.set_output(transform="pandas")

counts = y_train.value_counts(normalize=True)
cw = "balanced" if counts.min() < 0.10 else None
clf = HistGradientBoostingClassifier(random_state=seed, class_weight=cw) if cw else HistGradientBoostingClassifier(random_state=seed)

pipe = Pipeline([("pre", pre), ("clf", clf)])
if width <= 300:
    pipe.set_output(transform="pandas")

mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
mlca_runtime.save_model(pipe, "alternative_model.pkl")

print(f"train rows={n} features={len(feature_cols)} num={len(num_cols)} cat={len(cat_cols)}")
print(f"encoded width={width} class_weight={cw}")
print("saved alternative_model.pkl")
