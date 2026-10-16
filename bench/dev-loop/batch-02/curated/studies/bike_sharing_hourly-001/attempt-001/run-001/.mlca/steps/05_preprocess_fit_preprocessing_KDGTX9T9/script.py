import json
from itertools import combinations
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in proto:
        raise KeyError(f"MLCA: missing protocol key {key}; available keys: {sorted(proto)}")
train = pd.read_parquet(inputs["train"])
targets = proto["targets"]
targets = [targets] if isinstance(targets, str) else list(targets)
excluded = proto["excluded_columns"]
excluded = [excluded] if isinstance(excluded, str) else list(excluded)
group = proto.get("group_column")
task = proto["task"]
time_cols = set(task.get("time_columns", []))

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)
features = [c for c in train.columns if c not in drop]
key_like = []
for c in features:
    s = train[c]
    if s.nunique(dropna=False) == len(s) and not pd.api.types.is_float_dtype(s.dtype):
        if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or pd.api.types.is_integer_dtype(s.dtype):
            key_like.append(c)
features = [c for c in features if c not in key_like]

X = train[features].copy()
for c in list(features):
    if c in time_cols and (pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_string_dtype(X[c])):
        d = pd.to_datetime(X[c], errors="coerce")
        if d.notna().any():
            for p, s in {"year": d.dt.year, "month": d.dt.month, "day": d.dt.day, "dayofweek": d.dt.dayofweek}.items():
                new = f"{c}_{p}"
                X[new] = s
                features.append(new)
            features.remove(c)
            X.drop(columns=[c], inplace=True)

y = train[targets[0]]
num_screen = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
derived = {}
for c in num_screen:
    if X[c].equals(y):
        derived[c] = "exact target copy"
for r in (2, 3):
    for combo in combinations(num_screen, r):
        if any(c in derived for c in combo):
            continue
        sub = X[list(combo)]
        if (sub >= 0).all().all() and sub.sum(axis=1).equals(y):
            for c in combo:
                derived[c] = f"row sum {combo} equals target"
if derived:
    print("excluded derivations: " + "; ".join(f"{c} ({why})" for c, why in derived.items()))
features = [c for c in features if c not in derived]
X = X[features]

warn = []
if pd.api.types.is_numeric_dtype(y):
    for c in features:
        if pd.api.types.is_numeric_dtype(X[c]):
            corr = X[c].corr(y)
            if pd.notna(corr) and abs(corr) >= 0.98:
                warn.append(c)
if warn:
    print(f"correlation >=0.98 with target (kept): {warn}")

ph_cols = []
for c in X.select_dtypes(include=["object", "string"]).columns:
    s = X[c].astype("string").str.strip()
    if s.isin(["?", "-", "NA", "N/A"]).any() or s.eq("").any():
        ph_cols.append(c)
if ph_cols:
    print(f"placeholder tokens in {len(ph_cols)} columns: {ph_cols[:5]}")

numeric_cols, cat_cols, high_card = [], [], []
for c in features:
    s = X[c]
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or str(s.dtype) == "category":
        if s.nunique(dropna=False) > max(5, 0.05 * len(X)):
            high_card.append(c)
        else:
            cat_cols.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) <= 20 and not set(s.dropna().unique()).issubset({0, 1}):
        cat_cols.append(c)
    else:
        numeric_cols.append(c)
if high_card:
    print(f"dropped high-cardinality text/code columns: {high_card[:5]}")

features = numeric_cols + cat_cols
X = X[features]
width = len(numeric_cols)
for c in cat_cols:
    width += X[c].nunique(dropna=False)
sparse_out = width > 300

num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_out))])
pre = ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")
if not sparse_out:
    pre.set_output(transform="pandas")

pre = mlca_runtime.fit_record(pre, X[features], "train")
mlca_runtime.save_model(pre, "preprocessor.pkl")
mlca_runtime.record_metric("feature_count", len(features), "train")
mlca_runtime.record_metric("encoded_width", width, "train")
print(f"fitted preprocessor on {len(features)} features; encoded width ~{width}")
