import mlca_runtime
import json
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(Path(inputs["_protocol"]).read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in protocol:
        raise ValueError(f"Missing protocol key '{k}'; available: {list(protocol.keys())}")
targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group_col = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(inputs["train"])
n = len(train)

def is_text(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)

ph_tokens = ["?", "-", "NA", "N/A"]
ph_cols = []
for c in train.columns:
    if is_text(train[c]):
        ss = train[c].astype("string").str.strip()
        mask = ss.isin(ph_tokens) | (ss == "")
        if int(mask.sum()) > 0:
            ph_cols.append(c)
            cleaned = ss.where(~mask, pd.NA)
            train[c] = cleaned.where(train[c].notna(), np.nan).astype(object)
print(f"placeholder cols: {len(ph_cols)}")

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
key_drop = []
for c in feature_cols:
    s = train[c]
    nu = s.nunique(dropna=True)
    if is_text(s) and nu >= n - 1:
        key_drop.append(c)
    elif pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s) and nu >= n - 1 and nu > 100:
        key_drop.append(c)
feature_cols = [c for c in feature_cols if c not in key_drop]

X = train[feature_cols].copy()
y = train[target]

derived = set()
num_all = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c]) and X[c].notna().all() and (X[c] >= 0).all()]
for k in (1, 2, 3):
    pops = [c for c in num_all if c not in derived]
    for combo in combinations(pops, k):
        if (X[list(combo)].sum(axis=1) == y).all():
            derived.update(combo)
feature_cols = [c for c in feature_cols if c not in derived]
if derived:
    print(f"excluded derivation of {target}: {sorted(derived)}")

if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_categorical_dtype(y):
    yf = y.astype(float).values
    warns = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(X[c]):
            xv = X[c].astype(float).values
            v = ~np.isnan(xv) & ~np.isnan(yf)
            if v.sum() > 1 and abs(np.corrcoef(xv[v], yf[v])[0, 1]) >= 0.98:
                warns.append(c)
    if warns:
        print(f"keep high-corr (>=0.98): {warns}")

num_feats, cat_feats, drop_feats = [], [], []
for c in feature_cols:
    s = X[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        drop_feats.append(c)
    elif is_text(s):
        nu = s.nunique(dropna=True)
        if nu > max(20, int(0.05 * n)):
            drop_feats.append(c)
        else:
            cat_feats.append(c)
    elif pd.api.types.is_integer_dtype(s):
        vals = set(s.dropna().unique())
        if vals.issubset({0, 1}):
            num_feats.append(c)
        elif s.nunique(dropna=True) <= 15:
            cat_feats.append(c)
        else:
            num_feats.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        num_feats.append(c)
    else:
        drop_feats.append(c)
if drop_feats:
    print(f"dropped high-card/datetime: {drop_feats}")

num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
transformers = []
if num_feats:
    transformers.append(("num", num_pipe, num_feats))
if cat_feats:
    transformers.append(("cat", cat_pipe, cat_feats))
preproc = ColumnTransformer(transformers, remainder="drop")
preproc.set_output(transform="pandas")

X_train = train[num_feats + cat_feats].copy()
mlca_runtime.fit_record(preproc, X_train, "train", y=y)
mlca_runtime.save_model(preproc, "preprocessor.pkl")
mlca_runtime.record_fact_candidate("n_features_fit", len(num_feats) + len(cat_feats), source="preprocessor.pkl")
print(f"fitted: rows={len(X_train)} num={len(num_feats)} cat={len(cat_feats)}")
