import json
import mlca_runtime
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
ins = mlca_runtime.inputs()
proto = json.loads(ins["_protocol"].read_text())
if "targets" not in proto or "excluded_columns" not in proto:
    raise KeyError(f"protocol keys: {list(proto.keys())}")
train = pd.read_parquet(ins["train"])

targets = list(proto.get("targets", []) or [])
excluded = set(proto.get("excluded_columns", []) or [])
group = proto.get("group_column")
if group:
    excluded.add(group)


def keylike(s):
    if pd.api.types.is_numeric_dtype(s) and s.notna().all() and s.nunique() == len(s):
        return True
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) \
            and s.nunique(dropna=True) == len(s):
        return True
    return False


feat_cols, keys = [], []
for c in train.columns:
    if c in excluded or c in targets:
        continue
    (keys if keylike(train[c]) else feat_cols).append(c)
if keys:
    print(f"keylike dropped: {keys[:5]}")

deriv = []
for t in targets:
    if t not in train.columns:
        continue
    for c in feat_cols:
        if train[c].equals(train[t]):
            deriv.append(c)
            print(f"excluded {c}: equals target {t}")
feat_cols = [c for c in feat_cols if c not in deriv]

num_cols, cat_cols, dropped = [], [], []
n = len(train)
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        if s.dropna().nunique() <= 1:
            dropped.append(c)
        else:
            num_cols.append(c)
    elif pd.api.types.is_datetime64_any_dtype(s):
        dropped.append(c)
    else:
        u = s.nunique(dropna=True)
        if u > 100 or (u > 0.05 * n):
            dropped.append(c)
        else:
            cat_cols.append(c)
if dropped:
    print(f"dropped high-card/time/const: {dropped[:5]}")

ph_tokens = {"?", "-", "NA", "N/A", ""}
ph_cols = 0
for c in cat_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        z = s.astype(str).str.strip()
        if (z.isin(ph_tokens - {""}) | (z == "")).any():
            ph_cols += 1
if ph_cols:
    print(f"placeholder tokens found in {ph_cols} column(s)")

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore",
                                          sparse_output=False))])
pre = ColumnTransformer([("num", num_pipe, num_cols),
                         ("cat", cat_pipe, cat_cols)],
                        remainder="drop")

width = len(num_cols) + sum(int(train[c].nunique(dropna=True)) for c in cat_cols)
if width <= 300:
    pre.set_output(transform="pandas")

X = train[num_cols + cat_cols]
fitted = mlca_runtime.fit_record(pre, X, partition="train")
mlca_runtime.save_model(fitted, "preprocessor.pkl")
mlca_runtime.record_metric("n_features_raw", float(X.shape[1]), "train")
print(f"fitted preprocessor: rows={X.shape[0]} features={X.shape[1]}")
