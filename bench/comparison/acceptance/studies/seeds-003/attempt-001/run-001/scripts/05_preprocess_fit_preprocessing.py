import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
train = pd.read_parquet(paths["train"])

targets = list(protocol.get("targets") or [])
excluded = list(protocol.get("excluded_columns") or [])
group = protocol.get("group_column")

drop = set(targets) | set(excluded)
if group:
    drop.add(group)

feature_cols = sorted([c for c in train.columns if c not in drop])
keylike = []
for c in list(feature_cols):
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.notna().all():
        if s.nunique() == len(s):
            keylike.append(c)
feature_cols = [c for c in feature_cols if c not in keylike]
if keylike:
    print(f"excluded key-like: {keylike}")

phs = {"?", "-", "NA", "N/A", "na", "n/a", "#N/A", ""}
ph_cols = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        st = s.astype(str).str.strip()
        bad = st.isin(phs) | (st == "")
        if bad.any():
            ph_cols.append(c)
if ph_cols:
    print(f"placeholders in cols: {ph_cols}")

numeric_cols, cat_cols = [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    elif s.nunique() <= max(2, int(0.05 * len(train))):
        cat_cols.append(c)

log_cols = []
for c in numeric_cols:
    s = train[c].dropna()
    if len(s) > 10 and (s >= 0).all() and float(s.skew()) > 1.5:
        log_cols.append(c)

plain_num = [c for c in numeric_cols if c not in log_cols]

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())])
log_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                     ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

tfms = []
if plain_num: tfms.append(("num", num_pipe, plain_num))
if log_cols: tfms.append(("numlog", log_pipe, log_cols))
if cat_cols: tfms.append(("cat", cat_pipe, cat_cols))

pre = ColumnTransformer(tfms, remainder="drop")
pre.set_output(transform="pandas")
mlca_runtime.fit_record(pre, train[feature_cols], "train")
mlca_runtime.save_model(pre, "preprocessor.pkl")

probe = pre.transform(train[feature_cols].head(2))
print(f"features={len(feature_cols)} log1p={log_cols} output_cols={probe.shape[1]}")
