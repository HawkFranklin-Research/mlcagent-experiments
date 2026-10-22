import json
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in proto]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(proto)}")

targets = list(proto["targets"])
excluded = list(proto["excluded_columns"] or [])
group = proto["group_column"]
task = proto["task"]

train = pd.read_parquet(paths["train"])
for c in targets:
    if c not in train.columns:
        raise KeyError(f"Target {c!r} missing from train; columns: {list(train.columns)}")
y = train[targets].copy()

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)
feat_cols = [c for c in train.columns if c not in drop]

n = len(train)
key_like = set()
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if n and s.nunique(dropna=True) == n:
            key_like.add(c)
    elif pd.api.types.is_integer_dtype(s):
        if n > 1 and s.nunique(dropna=True) == n:
            key_like.add(c)
feat_cols = [c for c in feat_cols if c not in key_like]

numeric_cols, cat_cols, drop_cols = [], [], []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        nun = s.nunique(dropna=True)
        if n and nun <= max(20, int(0.05 * n)):
            cat_cols.append(c)
        else:
            drop_cols.append(c)
    else:
        drop_cols.append(c)

width = len(numeric_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
dense = width <= 300

transformers = []
if numeric_cols:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    if dense:
        num_pipe.set_output(transform="pandas")
    transformers.append(("num", num_pipe, numeric_cols))

if cat_cols:
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
    ])
    if dense:
        cat_pipe.set_output(transform="pandas")
    transformers.append(("cat", cat_pipe, cat_cols))

pre = ColumnTransformer(transformers, remainder="drop")
if dense:
    pre.set_output(transform="pandas")

pipe = Pipeline([("pre", pre), ("model", LinearRegression())])
if dense:
    pipe.set_output(transform="pandas")

X = train[feat_cols]
model = mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(model, "model.pkl")
print(f"train rows={len(train)} features={len(feat_cols)} numeric={len(numeric_cols)} categorical={len(cat_cols)}")
