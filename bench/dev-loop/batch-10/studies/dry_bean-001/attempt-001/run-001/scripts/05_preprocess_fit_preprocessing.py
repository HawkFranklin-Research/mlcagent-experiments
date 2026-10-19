import json
import itertools
import numpy as np
import pandas as pd
import mlca_runtime
from pandas.api.types import (
    is_numeric_dtype, is_datetime64_any_dtype, is_object_dtype,
    is_string_dtype, is_integer_dtype,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
protocol = json.loads(paths["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {sorted(d.keys())}")
    return d[k]

targets = list(need(protocol, "targets"))
excluded = list(need(protocol, "excluded_columns"))
task = need(protocol, "task")
group_col = protocol.get("group_column")
if not targets:
    raise ValueError("No targets declared")

def is_text_like(s):
    return is_object_dtype(s) or is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)

def is_key_like(s):
    n = len(s)
    if is_integer_dtype(s.dtype) and s.notna().all() and s.is_unique:
        vals = np.sort(s.unique())
        if len(vals) == n and np.all(np.diff(vals) == 1):
            return True
    if is_text_like(s):
        return s.nunique(dropna=False) == n
    return False

exclude_cols = set(excluded) | set(targets)
if group_col:
    exclude_cols.add(group_col)
feature_cols = [c for c in train.columns if c not in exclude_cols and not is_key_like(train[c])]

def same_series(a, b):
    if is_numeric_dtype(a) and is_numeric_dtype(b):
        return a.fillna(0).eq(b.fillna(0)).all() and a.isna().equals(b.isna())
    aa = a.astype("string").fillna("__NA__")
    bb = b.astype("string").fillna("__NA__")
    return aa.eq(bb).all()

derived = {}
for tgt in targets:
    if tgt not in train:
        continue
    y = train[tgt]
    for c in feature_cols:
        if c not in derived and same_series(train[c], y):
            derived[c] = f"equals {tgt}"
    if is_numeric_dtype(y):
        nums = [
            c for c in feature_cols
            if c not in derived and is_numeric_dtype(train[c])
            and train[c].notna().all() and train[c].ge(0).all()
        ]
        for k in (2, 3):
            for combo in itertools.combinations(nums, k):
                if any(c in derived for c in combo):
                    continue
                if np.allclose(train[list(combo)].sum(axis=1).values, y.values, rtol=0, atol=1e-9):
                    for c in combo:
                        derived[c] = f"row sum equals {tgt}"
if derived:
    print("target-derived excluded: " + ", ".join(f"{c} ({r})" for c, r in derived.items()))
feature_cols = [c for c in feature_cols if c not in derived]

ph_cols = []
for c in feature_cols:
    if is_text_like(train[c]):
        s = train[c].astype("string")
        stripped = s.str.strip()
        mask = stripped.isin(["?", "-", "NA", "N/A"]) | (s.notna() & (stripped == ""))
        if mask.fillna(False).any():
            ph_cols.append(c)
if ph_cols:
    for df in (train, val):
        for c in ph_cols:
            if c in df:
                s = df[c].astype("string").str.strip()
                s = s.mask(s.isin(["?", "-", "NA", "N/A"]) | (s == ""), np.nan)
                df[c] = s
    print(f"placeholders in {len(ph_cols)} text columns: {', '.join(ph_cols[:5])}")

num_cols, cat_cols, dropped = [], [], []
n_train = len(train)
for c in feature_cols:
    s = train[c]
    if is_datetime64_any_dtype(s):
        dropped.append(c)
    elif is_numeric_dtype(s):
        num_cols.append(c)
    else:
        nunique = s.nunique(dropna=True)
        if nunique > 0.05 * n_train:
            dropped.append(c)
        else:
            cat_cols.append(c)
if dropped:
    print("dropped high-card/datetime features: " + ", ".join(dropped[:10]))
feature_cols = num_cols + cat_cols
if not feature_cols:
    raise ValueError("No feature columns remain after screening")

cat_width = sum(train[c].nunique(dropna=True) for c in cat_cols)
n_encoded = len(num_cols) + cat_width
sparse = n_encoded > 300
transformers = []
if num_cols:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)),
    ])
    transformers.append(("cat", cat_pipe, cat_cols))

preprocessor = ColumnTransformer(transformers, remainder="drop")
if not sparse:
    if num_cols:
        num_pipe.set_output(transform="pandas")
    if cat_cols:
        cat_pipe.set_output(transform="pandas")
    preprocessor.set_output(transform="pandas")

X_train = train.loc[:, feature_cols].copy()
X_val = val.loc[:, feature_cols].copy()
mlca_runtime.fit_record(preprocessor, X_train, "train")
Xt_val = preprocessor.transform(X_val)
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

mlca_runtime.record_metric("n_train_rows", float(len(train)), "train")
mlca_runtime.record_metric("n_val_rows", float(len(val)), "val")
mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_val_transformed_columns", float(Xt_val.shape[1]), "val")
print(f"train {train.shape}, val {val.shape}; X_val transformed {Xt_val.shape}; features {len(feature_cols)}")
