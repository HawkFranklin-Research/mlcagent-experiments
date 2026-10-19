import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group_col = protocol.get("group_column")
target = targets[0]

blocked = set(targets) | set(excluded)
if group_col:
    blocked.add(group_col)
n = len(train)

feature_cols = []
for c in train.columns:
    if c in blocked:
        continue
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s.dtype) or pd.api.types.is_integer_dtype(s.dtype):
        if s.nunique(dropna=True) == n:
            continue
    feature_cols.append(c)

tgt_str = train[target].astype(str)
for c in list(feature_cols):
    try:
        if train[c].astype(str).equals(tgt_str):
            print(f"excluded {c}: exact match with target")
            feature_cols.remove(c)
    except Exception:
        pass

cat_cols = [c for c in feature_cols if train[c].dtype == object or pd.api.types.is_string_dtype(train[c].dtype)]
num_cols = [c for c in feature_cols if c not in cat_cols]

for c in list(cat_cols):
    if train[c].nunique(dropna=True) > 0.05 * n:
        cat_cols.remove(c)
        feature_cols.remove(c)
        print(f"dropped high-cardinality text col {c}")

pats = ['?', '-', 'NA', 'N/A', '']
found_cols = 0
for c in cat_cols:
    for df in (train, val):
        s = df[c]
        mask = s.notna()
        cleaned = s.astype(str).str.strip().where(mask, np.nan)
        if df is train and cleaned.isin(pats).any():
            found_cols += 1
        df[c] = cleaned.replace(pats, np.nan)
if found_cols:
    print(f"placeholder tokens ('?','-','NA','N/A','') in {found_cols} cat cols -> NaN")

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
ct = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")
ct.set_output(transform="pandas")

X_tr = train[feature_cols]
X_val = val[feature_cols]
ct = mlca_runtime.fit_record(ct, X_tr, "train")
Xt = ct.transform(X_val)
print(f"train={X_tr.shape}, val_transformed={Xt.shape}, features={len(feature_cols)}")

mlca_runtime.save_model(ct, "preprocessor.pkl")
