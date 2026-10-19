import json
import itertools
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"missing protocol key {key!r}; available keys: {list(protocol)}")

target_cols = list(protocol["targets"])
target = target_cols[0]
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")
time_cols = protocol["task"].get("time_columns", [])
if target not in train.columns:
    raise KeyError(f"target {target!r} not in training columns")

placeholder_count = 0
for col in train.select_dtypes(include=["object", "string"]).columns:
    s = train[col].astype("string")
    stripped = s.str.strip()
    mask = stripped.isna() | stripped.isin(["?", "-", "NA", "N/A", ""])
    if mask.any():
        placeholder_count += 1
    train[col] = stripped.where(~mask, np.nan)
print(f"placeholder/whitespace cleaned columns: {placeholder_count}")

df = train.copy()
date_cols = []
for col in df.columns:
    if col in target_cols or col in excluded or col == group_col:
        continue
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        date_cols.append(col)
    elif col in time_cols and not pd.api.types.is_numeric_dtype(df[col]):
        date_cols.append(col)
for col in date_cols:
    dt = pd.to_datetime(df[col], errors="coerce")
    if dt.notna().mean() > 0.8:
        df[col + "_year"] = dt.dt.year
        df[col + "_month"] = dt.dt.month
        df[col + "_day"] = dt.dt.day
        df[col + "_dayofweek"] = dt.dt.dayofweek
        if (dt.dt.hour != 0).any():
            df[col + "_hour"] = dt.dt.hour
        df = df.drop(columns=[col])

candidate = [c for c in df.columns if c not in target_cols and c not in excluded and c != group_col]
key_like = []
for c in list(candidate):
    s = df[c]
    if s.isna().any() or len(s) <= 1:
        continue
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == len(s):
        key_like.append(c)
        candidate.remove(c)
    elif pd.api.types.is_integer_dtype(s) and not set(s.dropna().unique()).issubset({0, 1}):
        if s.nunique(dropna=True) == len(s):
            key_like.append(c)
            candidate.remove(c)

y = df[target]
derived = []
if pd.api.types.is_numeric_dtype(y):
    for c in list(candidate):
        if pd.api.types.is_numeric_dtype(df[c]) and np.array_equal(df[c].to_numpy(), y.to_numpy()):
            derived.append((c, "equals target"))
            candidate.remove(c)
    numeric = [c for c in candidate if pd.api.types.is_numeric_dtype(df[c]) and (df[c].dropna() >= 0).all()]
    for r in (2, 3):
        for combo in itertools.combinations(numeric, r):
            if all(c in candidate for c in combo):
                s = df[list(combo)].sum(axis=1, skipna=False)
                if np.array_equal(s.to_numpy(), y.to_numpy()):
                    for c in combo:
                        if c in candidate:
                            candidate.remove(c)
                    derived.append((combo, "sum equals target"))
                    break
if derived:
    print("excluded derivations: " + "; ".join(f"{c} ({r})" for c, r in derived))

warn = []
if pd.api.types.is_numeric_dtype(y):
    for c in candidate:
        if pd.api.types.is_numeric_dtype(df[c]):
            corr = df[c].corr(y)
            if abs(corr) >= 0.98:
                warn.append(c)
if warn:
    print("high-correlation kept: " + ", ".join(warn))

num_cols, cat_cols, high_drop = [], [], []
n = len(df)
for c in candidate:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=True) > max(5, 0.05 * n):
            high_drop.append(c)
        else:
            cat_cols.append(c)
if high_drop:
    print("dropped high-cardinality text: " + ", ".join(high_drop))

feature_cols = num_cols + cat_cols
X = df[feature_cols]
width = len(num_cols) + sum(df[c].nunique(dropna=True) for c in cat_cols)
sparse_out = width > 300
transformers = []
if num_cols:
    transformers.append(("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), num_cols))
if cat_cols:
    transformers.append(("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_out)),
    ]), cat_cols))
preprocessor = ColumnTransformer(transformers, remainder="drop")
if not sparse_out:
    preprocessor.set_output(transform="pandas")

mlca_runtime.fit_record(preprocessor, X, "train", y=None)
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
print(f"fitted preprocessor: {len(feature_cols)} features -> {width} encoded columns")
