import json
import numpy as np
import pandas as pd
import mlca_runtime
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {list(d.keys())}")
    return d[k]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group = need(protocol, "group_column")
target = targets[0]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
y = train[target]

text_cols = [
    c for c in train.columns
    if c not in set(targets) | set(excluded) | ({group} if group is not None else set())
    and (pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]))
]
placeholders = {"?", "-", "NA", "N/A", ""}
found = []
for c in text_cols:
    s = train[c].astype("string").str.strip()
    if s.isin(placeholders).any():
        found.append(c)
        train[c] = s.replace(list(placeholders), np.nan)
        val[c] = val[c].astype("string").str.strip().replace(list(placeholders), np.nan)
print(f"Placeholder columns: {len(found)} {found}" if found else "No placeholder tokens found")

drop_cols = set(targets) | set(excluded)
if group is not None:
    drop_cols.add(group)

base = []
for c in train.columns:
    if c in drop_cols:
        continue
    s = train[c]
    all_distinct = s.nunique(dropna=False) == len(s)
    name = str(c).lower()
    if all_distinct and not pd.api.types.is_float_dtype(s) and (
        pd.api.types.is_object_dtype(s) or "id" in name or name.endswith("_id")
    ):
        continue
    base.append(c)

num_cols, cat_cols = [], []
for c in list(base):
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        for part, fn in [("dayofweek", lambda x: x.dt.dayofweek), ("month", lambda x: x.dt.month)]:
            train[f"{c}_{part}"] = fn(s)
            val[f"{c}_{part}"] = fn(val[c])
            num_cols.append(f"{c}_{part}")
        base.remove(c)
    elif pd.api.types.is_bool_dtype(s):
        cat_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        uniq = s.nunique(dropna=True)
        if pd.api.types.is_integer_dtype(s) and uniq <= 15 and not set(s.dropna().unique()).issubset({0, 1}):
            cat_cols.append(c)
        else:
            num_cols.append(c)
    else:
        uniq = s.nunique(dropna=True)
        if uniq > max(20, 0.05 * len(train)):
            base.remove(c)
        else:
            cat_cols.append(c)

features = num_cols + cat_cols
dropped_derived = []
if pd.api.types.is_numeric_dtype(y):
    y_num = pd.to_numeric(y, errors="coerce")
    for c in list(features):
        if (pd.to_numeric(train[c], errors="coerce") == y_num).all():
            features.remove(c)
            dropped_derived.append((c, "exact target equality"))
    cand = [c for c in features if c in num_cols and (train[c].dropna() >= 0).all()]
    for r in (1, 2, 3):
        for combo in combinations(cand, r):
            if any(c not in features for c in combo):
                continue
            if (train[list(combo)].sum(axis=1) == y_num).all():
                for c in combo:
                    if c in features:
                        features.remove(c)
                        dropped_derived.append((c, f"row sum {combo} equals target"))
                break
    if dropped_derived:
        print("Excluded derived features: " + "; ".join(f"{c} ({r})" for c, r in dropped_derived))
    warn = [c for c in features if c in num_cols and abs(train[c].corr(y_num)) >= 0.98]
    if warn:
        print(f"Correlation >=0.98 with target (kept): {warn}")

num_cols = [c for c in features if c in num_cols]
cat_cols = [c for c in features if c in cat_cols]
transformers = []
if num_cols:
    transformers.append(("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), num_cols))
if cat_cols:
    transformers.append(("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), cat_cols))

pre = ColumnTransformer(transformers, remainder="drop")
width = len(num_cols) + sum(train[c].nunique(dropna=True) + 1 for c in cat_cols)
if width <= 300:
    pre.set_output(transform="pandas")

fitted = mlca_runtime.fit_record(pre, train[features], "train")
Xt_val = fitted.transform(val[features])
print(f"Features: {len(features)}; encoded width: {Xt_val.shape[1]}; val rows: {Xt_val.shape[0]}")
mlca_runtime.save_model(fitted, "preprocessor.pkl")
