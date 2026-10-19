import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
try:
    targets = list(protocol["targets"])
    excluded = list(protocol["excluded_columns"])
    task = protocol["task"]
    group = protocol["group_column"]
except KeyError as e:
    raise KeyError(f"Missing {e.args[0]} in _protocol; available keys: {sorted(protocol)}")

train = pd.read_parquet(inp["train"])
drop = set(targets) | set(excluded) | {group}
feat_cols = [c for c in train.columns if c not in drop]

keylike = []
for c in list(feat_cols):
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if len(s) > 1 and s.nunique(dropna=True) == s.notna().sum():
            keylike.append(c)
    elif pd.api.types.is_integer_dtype(s):
        name = c.lower()
        if len(s) > 1 and s.nunique(dropna=True) == s.notna().sum() and any(t in name for t in ("id", "no", "code", "key", "index")):
            keylike.append(c)
feat_cols = [c for c in feat_cols if c not in set(keylike)]
if not feat_cols:
    raise ValueError("No feature columns after excluding targets, exclusions, group and key-like columns")

placeholders = ["?", "-", "NA", "N/A", ""]
bad_cols = 0
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.astype("string").str.strip().isin(placeholders).any():
            bad_cols += 1
if bad_cols:
    print(f"placeholder tokens found in {bad_cols} columns")

num_cols, cat_cols, date_cols = [], [], []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        cat_cols.append(c)

cat_limit = max(20, min(50, int(0.05 * len(train))))
cat_keep, high_card = [], []
for c in cat_cols:
    if train[c].nunique(dropna=True) <= cat_limit:
        cat_keep.append(c)
    else:
        high_card.append(c)
cat_cols = cat_keep

heavy, plain = [], []
for c in num_cols:
    s = train[c].dropna()
    if len(s) > 2 and (s >= 0).all() and s.nunique() > 2 and s.skew() > 1:
        heavy.append(c)
    else:
        plain.append(c)

width = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
pandas_out = width <= 300
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=not pandas_out)

def num_pipe(use_log):
    steps = []
    if use_log:
        steps.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    steps += [("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    return Pipeline(steps)

def cat_pipe():
    steps = [(f"imp_{i}", SimpleImputer(missing_values=tok, strategy="most_frequent")) for i, tok in enumerate(placeholders)]
    steps += [("imp", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)]
    return Pipeline(steps)

transformers = []
if heavy:
    transformers.append(("num_heavy", num_pipe(True), heavy))
if plain:
    transformers.append(("num_plain", num_pipe(False), plain))
if cat_cols:
    transformers.append(("cat", cat_pipe(), cat_cols))

ct = ColumnTransformer(transformers, remainder="drop")
if pandas_out:
    for _, tr, _ in transformers:
        tr.set_output(transform="pandas")
    ct.set_output(transform="pandas")

feat_cols = heavy + plain + cat_cols
X_train = train[feat_cols].copy()
fitted = mlca_runtime.fit_record(ct, X_train, "train")
mlca_runtime.save_model(fitted, "preprocessor.pkl")

print(f"features={len(feat_cols)} numeric={len(num_cols)} categorical={len(cat_cols)} dropped_keylike={len(keylike)}")
print(f"high_card_dropped={len(high_card)} date_dropped={len(date_cols)} pandas_out={pandas_out}")
