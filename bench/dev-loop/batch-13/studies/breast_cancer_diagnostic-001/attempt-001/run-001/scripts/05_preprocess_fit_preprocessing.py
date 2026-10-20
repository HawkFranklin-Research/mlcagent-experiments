import mlca_runtime
import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for k in ["targets", "excluded_columns", "task"]:
    if k not in protocol:
        raise ValueError(f"Missing key {k}; available: {list(protocol.keys())}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
task = protocol["task"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_keylike(s):
    if s.dtype == object:
        return len(s) > 1 and s.nunique() == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        return len(s) > 1 and s.nunique() == len(s)
    return False

feature_cols = [c for c in train.columns if c not in targets and c not in excluded]
if group:
    feature_cols = [c for c in feature_cols if c != group]
key_like = [c for c in feature_cols if is_keylike(train[c])]
if key_like:
    print(f"key-like columns dropped: {key_like}")
feature_cols = [c for c in feature_cols if c not in key_like]

target_col = targets[0]
y_train = train[target_col]
y_is_num = pd.api.types.is_numeric_dtype(y_train.dtype)
derived = []
for c in feature_cols:
    s = train[c]
    if y_is_num and pd.api.types.is_numeric_dtype(s.dtype):
        if (s == y_train).all():
            derived.append(c)
if y_is_num:
    yf = y_train.astype(float)
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train[c].dtype):
            try:
                corr = float(np.corrcoef(train[c].astype(float).to_numpy(), yf.to_numpy())[0, 1])
                if abs(corr) >= 0.98:
                    print(f"high-corr warning: {c} {corr:.3f}")
            except Exception:
                pass
for c in derived:
    print(f"excluded feature {c}: exact derivation of target")
feature_cols = [c for c in feature_cols if c not in derived]

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = []
for c in feature_cols:
    if train[c].dtype == object:
        vals = train[c].dropna().astype(str)
        if vals.str.strip().isin(placeholders).any() or (vals != vals.str.strip()).any():
            ph_cols.append(c)
print(f"placeholder-text columns: {len(ph_cols)}")

numeric_cols, cat_cols = [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        continue
    if pd.api.types.is_numeric_dtype(s.dtype):
        numeric_cols.append(c)
    else:
        cat_cols.append(c)

num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())])
transformers = [("num", num_pipe, numeric_cols)]
if cat_cols:
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    transformers.append(("cat", cat_pipe, cat_cols))

pre = ColumnTransformer(transformers, remainder="drop")

width = len(numeric_cols) + sum(max(1, train[c].nunique()) for c in cat_cols)
if width <= 300:
    pre.set_output(transform="pandas")

X_train = train[feature_cols]
X_val = val[feature_cols]
mlca_runtime.fit_record(pre, X_train, "train", y=y_train)
Xt = pre.transform(X_val)
print(f"features={X_train.shape[1]}, val_transformed={Xt.shape}")

mlca_runtime.save_model(pre, "preprocessor.pkl")
