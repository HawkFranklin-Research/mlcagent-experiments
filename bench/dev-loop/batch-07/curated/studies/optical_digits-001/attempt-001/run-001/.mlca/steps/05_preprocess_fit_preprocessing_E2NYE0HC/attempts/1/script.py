import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(Path(inputs["_protocol"]).read_text())

for key in ("targets", "excluded_columns"):
    if key not in protocol:
        raise KeyError(f"missing '{key}'; available top-level keys: {sorted(protocol)}")

targets = protocol["targets"] or []
excluded = protocol["excluded_columns"] or []
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)


def is_key_like(s):
    nonnull = s.dropna()
    if len(nonnull) == 0:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return nonnull.nunique() == len(nonnull)
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return nonnull.nunique() == len(nonnull) and nonnull.nunique() > 1
    return False


feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    if is_key_like(train[c]):
        continue
    feature_cols.append(c)

placeholders_found = set()
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        vals = s.dropna().astype(str)
        for tok in ("?", "-", "NA", "N/A"):
            if (vals.str.strip() == tok).any():
                placeholders_found.add(tok)

num_cols, cat_cols = [], []
n_rows = len(train)
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=True) > 0.05 * n_rows:
            continue
        cat_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        try:
            pd.to_datetime(s.dropna().iloc[:20])
            continue
        except Exception:
            cat_cols.append(c)

if placeholders_found:
    print(f"placeholder tokens found in text columns: {sorted(placeholders_found)}")

log_cols, plain_num = [], []
for c in num_cols:
    s = train[c].dropna()
    if len(s) and s.min() >= 0:
        try:
            if s.skew() > 2:
                log_cols.append(c)
                continue
        except Exception:
            pass
    plain_num.append(c)

est_width = len(plain_num) + len(log_cols)
for c in cat_cols:
    est_width += int(train[c].nunique(dropna=True))
wide = est_width > 300

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
log_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(np.log1p, validate=False)),
    ("scale", StandardScaler()),
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=wide)),
])

transformers = []
if plain_num:
    transformers.append(("num", num_pipe, plain_num))
if log_cols:
    transformers.append(("log", log_pipe, log_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))

pre = ColumnTransformer(transformers, remainder="drop")
if not wide:
    pre.set_output(transform="pandas")

pre.fit(train[feature_cols])
mlca_runtime.save_model(pre, "preprocessor.pkl")

mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
mlca_runtime.record_fact_candidate("preprocessor_encoded_width", float(est_width), source="preprocessor.pkl")

print(f"features={len(feature_cols)} numeric={len(plain_num)} log1p={len(log_cols)} categorical={len(cat_cols)}")
print(f"encoded_width={est_width} wide={wide} saved=preprocessor.pkl")
