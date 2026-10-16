import mlca_runtime
import json
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(Path(inputs["_protocol"]).read_text())

for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing key {k}. Available top-level keys: {list(protocol.keys())}")

targets = protocol["targets"]
if not isinstance(targets, list):
    targets = [targets]
excluded = list(protocol["excluded_columns"])
group_col = protocol["group_column"]
task = protocol["task"]

n = len(train)
drop_set = set(targets) | set(excluded)
if group_col is not None:
    drop_set.add(group_col)

def is_key_like(c):
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=False) == n and n > 1:
            return True
    if pd.api.types.is_integer_dtype(s) and not set(s.dropna().unique()) <= {0, 1}:
        vals = np.sort(s.dropna().unique())
        if len(vals) == n and n > 1 and np.array_equal(vals, np.arange(vals[0], vals[0] + len(vals))):
            return True
    return False

feature_cols = [c for c in train.columns if c not in drop_set and not is_key_like(c)]
excluded_deriv = set()
deriv_msgs = []

for target in targets:
    if target not in train.columns:
        continue
    y = train[target]
    for f in feature_cols:
        if pd.api.types.is_numeric_dtype(train[f]) and np.allclose(train[f], y, rtol=0, atol=1e-9):
            excluded_deriv.add(f)
            deriv_msgs.append(f"{f}={target}")
    num_feats = [f for f in feature_cols if pd.api.types.is_numeric_dtype(train[f]) and (train[f].dropna() >= 0).all()]
    for r in (1, 2, 3):
        for combo in combinations(num_feats, r):
            if any(c in excluded_deriv for c in combo):
                continue
            if np.allclose(train[list(combo)].sum(axis=1), y, rtol=0, atol=1e-9):
                excluded_deriv.update(combo)
                deriv_msgs.append(f"{'+'.join(combo)}={target}")

if deriv_msgs:
    print("excluded derivations: " + "; ".join(deriv_msgs))
feature_cols = [c for c in feature_cols if c not in excluded_deriv]

warn = []
for target in targets:
    if target in train.columns and pd.api.types.is_numeric_dtype(train[target]):
        for f in feature_cols:
            if pd.api.types.is_numeric_dtype(train[f]) and train[f].nunique() > 1 and train[target].nunique() > 1:
                corr = np.corrcoef(train[f], train[target])[0, 1]
                if abs(corr) >= 0.98:
                    warn.append(f)
if warn:
    print("correlation >=0.98 with target (kept): " + ", ".join(sorted(set(warn))))

placeholders = {"?", "-", "NA", "N/A"}
ph_cols = []
for c in feature_cols:
    if not pd.api.types.is_numeric_dtype(train[c]):
        s = train[c].astype(str)
        found = (s.str.strip() == "").any() or any((s == p).any() for p in placeholders)
        if found:
            ph_cols.append(c)
if ph_cols:
    print(f"placeholder tokens in {len(ph_cols)} columns: {', '.join(ph_cols)}")

numeric_cols, cat_cols, drop_cols = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    elif s.nunique(dropna=False) > 0.05 * n:
        drop_cols.append(c)
    else:
        cat_cols.append(c)
if drop_cols:
    print(f"dropped high-cardinality/text columns: {', '.join(drop_cols)}")

n_out = len(numeric_cols) + sum(train[c].nunique(dropna=False) for c in cat_cols)
sparse = n_out > 300
num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]) if numeric_cols else "drop"
cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse))]) if cat_cols else "drop"
pre = ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")

if n_out <= 300:
    pre.set_output(transform="pandas")
    if isinstance(num_pipe, Pipeline):
        num_pipe.set_output(transform="pandas")
    if isinstance(cat_pipe, Pipeline):
        cat_pipe.set_output(transform="pandas")

X = train[feature_cols].copy()
mlca_runtime.fit_record(pre, X, partition="train")
mlca_runtime.save_model(pre, "preprocessor.pkl")
mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_training_rows", float(n), "train")
print(f"fitted preprocessor on {n} rows, {len(feature_cols)} features")
