import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
targets = protocol["targets"]
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column", None)

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
candidates = [c for c in train.columns if c not in drop]

n = len(train)
key_like = []
for c in candidates:
    s = train[c]
    if (pd.api.types.is_integer_dtype(s) or pd.api.types.is_object_dtype(s)) \
            and s.nunique(dropna=False) == n:
        key_like.append(c)
feature_cols = [c for c in candidates if c not in key_like]

target_col = targets[0]
y = train[target_col]
derived = []
if pd.api.types.is_numeric_dtype(y) and len(feature_cols) <= 500:
    yv = y.astype(float).to_numpy()
    keep = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train[c]) and np.allclose(
            train[c].astype(float).to_numpy(), yv, equal_nan=True):
            derived.append(c)
            continue
        keep.append(c)
    feature_cols = keep

dt_cols = [c for c in feature_cols if pd.api.types.is_datetime64_any_dtype(train[c])]
ref = pd.Timestamp("1970-01-01")
for c in dt_cols:
    train[c + "_days"] = (train[c] - ref).dt.days
    val[c + "_days"] = (val[c] - ref).dt.days
feature_cols = [c for c in feature_cols if c not in dt_cols] + [c + "_days" for c in dt_cols]

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in feature_cols if c not in numeric_cols]

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = [c for c in cat_cols
           if train[c].astype(str).str.strip().isin(placeholders).any()]

transformers = []
if numeric_cols:
    transformers.append(("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]), numeric_cols))
if cat_cols:
    transformers.append(("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), cat_cols))

preproc = ColumnTransformer(transformers=transformers, remainder="drop")

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()
mlca_runtime.fit_record(preproc, X_train, "train", y=y)
Xt_val = preproc.transform(X_val)

mlca_runtime.record_metric("n_features", len(feature_cols), "train")
if key_like:
    print(f"Excluded {len(key_like)} key-like columns")
if derived:
    print(f"Excluded {len(derived)} derived columns")
print(f"Placeholder tokens in {len(ph_cols)} text columns")
print(f"Fit train {X_train.shape} val {X_val.shape} -> {Xt_val.shape}; num={len(numeric_cols)} cat={len(cat_cols)}")
mlca_runtime.save_model(preproc, "preprocessor.pkl")
