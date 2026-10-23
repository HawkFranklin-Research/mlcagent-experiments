import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns", "group_column", "task"):
    if k not in proto:
        raise KeyError(f"missing {k}; available: {list(proto)}")
targets, excluded, group, task = proto["targets"], proto["excluded_columns"], proto["group_column"], proto["task"]
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
if len(targets) != 1:
    raise ValueError("expected exactly one target")
target = targets[0]
y = train[target]
drop = set(targets) | set(excluded)
if group is not None:
    if group not in train.columns:
        raise KeyError(f"group column {group!r} missing")
    drop.add(group)
base = [c for c in train.columns if c not in drop]
n = len(train)
key_like = []
for c in base:
    s = train[c]
    if n > 1 and s.nunique(dropna=True) == n:
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            key_like.append(c)
base = [c for c in base if c not in key_like]
numeric_target = pd.api.types.is_numeric_dtype(y)
derived = []
for c in base:
    s = train[c]
    if numeric_target:
        if pd.api.types.is_numeric_dtype(s) and (s == y).all():
            derived.append(c)
    elif (s.astype("string") == y.astype("string")).all():
        derived.append(c)
base = [c for c in base if c not in derived]

ph_counts = {}
ph_tokens = set()
for c in base:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        st = s.astype("string").str.strip()
        for tok in ("?", "-", "NA", "N/A"):
            cnt = int((st == tok).sum())
            if cnt:
                ph_counts[tok] = ph_counts.get(tok, 0) + 1
                ph_tokens.add(tok)
        cnt = int((st == "").sum())
        if cnt:
            ph_counts["blank"] = ph_counts.get("blank", 0) + 1
            ph_tokens.add("")

num_cols, cat_cols, high_card = [], [], []
limit = max(20, int(0.05 * n))
for c in base:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        high_card.append(c)
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=True) <= limit:
            cat_cols.append(c)
        else:
            high_card.append(c)
    else:
        num_cols.append(c)
feature_cols = num_cols + cat_cols
if not feature_cols:
    raise ValueError("no feature columns remain")

width = len(num_cols) + sum(train[c].nunique(dropna=False) for c in cat_cols)
sparse = width > 300
num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
    ("scale", StandardScaler()),
])
cat_steps = []
for i, tok in enumerate(sorted(ph_tokens)):
    cat_steps.append((f"ph{i}", SimpleImputer(missing_values=tok, strategy="constant", fill_value=np.nan, keep_empty_features=True)))
cat_steps.append(("imp", SimpleImputer(strategy="most_frequent", keep_empty_features=True)))
cat_steps.append(("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse, dtype=np.float32)))
cat_pipe = Pipeline(cat_steps)
transformers = [("num", num_pipe, num_cols)]
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
ct = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
if not sparse and hasattr(ct, "set_output"):
    ct.set_output(transform="pandas")
pipe = Pipeline([("preprocess", ct)])
if not sparse and hasattr(pipe, "set_output"):
    pipe.set_output(transform="pandas")

X_train = train[feature_cols]
X_val = val[feature_cols]
pipe = mlca_runtime.fit_record(pipe, X_train, "train")
Xt = pipe.transform(X_val)
mlca_runtime.save_model(pipe, "preprocessor.pkl")
print("features:", len(feature_cols), "numeric:", len(num_cols), "categorical:", len(cat_cols), "key_like:", len(key_like), "high_card_dropped:", len(high_card))
print("excluded target-derived:", derived if derived else "none")
print("placeholders:", ph_counts if ph_counts else "none")
print("encoded width estimate:", width, "sparse:", sparse)
print("val transformed shape:", getattr(Xt, "shape", None))
