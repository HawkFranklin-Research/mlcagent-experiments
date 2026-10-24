import json
import numpy as np
import pandas as pd
import mlca_runtime
from pandas.api.types import (
    is_object_dtype, is_string_dtype, is_integer_dtype,
    is_numeric_dtype, is_datetime64_any_dtype,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]
train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

exclude_set = set(targets) | set(excluded)
if group is not None:
    exclude_set.add(group)
feature_cols = [c for c in train.columns if c not in exclude_set]

key_drop = []
keep = []
for c in feature_cols:
    s = train[c]
    is_cat = is_object_dtype(s) or is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)
    if is_cat and s.nunique(dropna=False) == len(s):
        key_drop.append(c)
    elif is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
        key_drop.append(c)
    else:
        keep.append(c)
feature_cols = keep

derived = []
for t in targets:
    if t not in train:
        continue
    y = train[t]
    for c in feature_cols:
        s = train[c]
        if is_numeric_dtype(s) and s.notna().all() and (s == y).all():
            derived.append(c)
if derived:
    feature_cols = [c for c in feature_cols if c not in set(derived)]

X_train = train.loc[:, feature_cols].copy()
X_val = val.loc[:, feature_cols].copy()

date_cols = [c for c in feature_cols if is_datetime64_any_dtype(X_train[c])]
derived_date = []
for c in date_cols:
    for part in ["year", "month", "day", "dayofweek"]:
        newc = f"{c}_{part}"
        X_train[newc] = getattr(X_train[c].dt, part)
        X_val[newc] = getattr(X_val[c].dt, part)
        derived_date.append(newc)
    X_train = X_train.drop(columns=[c])
    X_val = X_val.drop(columns=[c])
feature_cols = [c for c in feature_cols if c not in date_cols] + derived_date

text_cols = [c for c in feature_cols if is_object_dtype(X_train[c]) or is_string_dtype(X_train[c])]
ph_cols = []
for c in text_cols:
    s = X_train[c].dropna().astype(str).str.strip()
    if s.isin(["?", "-", "NA", "N/A", ""]).any():
        ph_cols.append(c)
for c in ph_cols:
    for frame in (X_train, X_val):
        frame[c] = frame[c].astype("object").str.strip().replace(
            {"?": np.nan, "-": np.nan, "NA": np.nan, "N/A": np.nan, "": np.nan}
        )

cat_all = [
    c for c in feature_cols
    if is_object_dtype(X_train[c]) or is_string_dtype(X_train[c]) or isinstance(X_train[c].dtype, pd.CategoricalDtype)
]
high_card = [c for c in cat_all if X_train[c].nunique(dropna=True) > max(5, 0.05 * len(X_train))]
if high_card:
    feature_cols = [c for c in feature_cols if c not in set(high_card)]
    X_train = X_train.drop(columns=high_card)
    X_val = X_val.drop(columns=high_card)

num_cols = [c for c in feature_cols if is_numeric_dtype(X_train[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]

corr_warn = []
for t in targets:
    if t in train and is_numeric_dtype(train[t]) and train[t].nunique() > 1 and num_cols:
        corr = X_train[num_cols].corrwith(train[t]).abs()
        corr_warn.extend(c for c, v in corr[corr >= 0.98].items())
corr_warn = list(dict.fromkeys(corr_warn))

encoded_width = len(num_cols) + sum(X_train[c].nunique(dropna=True) for c in cat_cols)
sparse = encoded_width > 300

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)),
])
transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
preprocessor = ColumnTransformer(
    transformers,
    remainder="drop",
    sparse_threshold=1.0 if sparse else 0.3,
)
if not sparse:
    preprocessor.set_output(transform="pandas")
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")

mlca_runtime.fit_record(preprocessor, X_train, "train")
val_trans = preprocessor.transform(X_val)
shape = getattr(val_trans, "shape", None)

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_excluded_keylike", float(len(key_drop)), "train")
mlca_runtime.record_metric("n_derived_excluded", float(len(derived)), "train")
mlca_runtime.record_metric("encoded_width", float(encoded_width), "train")

print(f"train rows={len(X_train)} features={len(feature_cols)} encoded_width={encoded_width} sparse={sparse}")
if derived:
    print("excluded target-derived: " + ", ".join(derived[:5]))
if corr_warn:
    print("corr>=0.98 kept: " + ", ".join(corr_warn[:5]))
if ph_cols or high_card:
    print("placeholder/high-cardinality: " + ", ".join((ph_cols + high_card)[:5]))
print(f"val transformed shape={shape}")
