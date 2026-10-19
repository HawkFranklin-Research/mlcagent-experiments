import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column") or None
task = protocol["task"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop_cols = set(targets) | set(excluded)
if group_col:
    drop_cols.add(group_col)
feats = [c for c in train.columns if c not in drop_cols]

# key-like columns: non-numeric with all distinct values
key_like = [c for c in feats if not pd.api.types.is_numeric_dtype(train[c])
            and train[c].nunique(dropna=True) == len(train)]
if key_like:
    print(f"dropping key-like columns: {key_like}")
    feats = [c for c in feats if c not in key_like]

# placeholder text handling
placeholders = {"?", "-", "NA", "N/A", ""}
ph_found = 0
for c in feats:
    if not pd.api.types.is_numeric_dtype(train[c]):
        vals = train[c].astype("object").where(train[c].notna(), None)
        s = pd.Series(vals).astype("string")
        s = s.str.strip()
        mask = s.isin(placeholders)
        if mask.any() or (s == "").any():
            ph_found += 1
        train[c] = s.mask(s.isin(placeholders), pd.NA).astype("object")
        vs = val[c].astype("string").str.strip()
        val[c] = vs.mask(vs.isin(placeholders), pd.NA).astype("object")
if ph_found:
    print(f"placeholder tokens in {ph_found} columns; converted to NaN")

# target-derivation screen on training rows
y_col = targets[0]
y = train[y_col]
to_exclude = {}
for c in feats:
    comparable = y.notna()
    eq = train.loc[comparable, c].fillna(object()).astype(str) == y.loc[comparable].astype(str)
    if eq.all():
        to_exclude[c] = "exact match with target"

num_frame = train[[c for c in feats if pd.api.types.is_numeric_dtype(train[c])]]
if not num_frame.empty and (num_frame.fillna(0).values >= 0).all():
    for combo in combinations(range(num_frame.shape[1]), 2):
        rs = num_frame.iloc[:, list(combo)].fillna(0).sum(axis=1)
        if (rs.astype(float) == y.values.astype(float)).all():
            for i in combo:
                col = num_frame.columns[i]
                to_exclude.setdefault(col, "row-sum equals target")
            break

for c, reason in to_exclude.items():
    print(f"excluding {c}: {reason}")
feats = [c for c in feats if c not in to_exclude]

# numeric correlations warning
if pd.api.types.is_numeric_dtype(y) and task.get("type") == "binary":
    num_keep = [c for c in feats if pd.api.types.is_numeric_dtype(train[c])]
    if num_keep:
        corrs = train[num_keep].corrwith(y).abs().dropna()
        hi = corrs[corrs >= 0.98].index.tolist()
        if hi:
            print(f"high-corr with target (kept): {hi}")

# split numeric / categorical / high-cardinality
n = len(train)
num_cols, cat_cols, drop_hi = [], [], []
for c in feats:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) or (s.dropna().isin([0, 1]).all() and s.notna().sum() > 0):
        num_cols.append(c)
    else:
        nun = s.nunique(dropna=True)
        if nun > 0.05 * n:
            drop_hi.append(c)
        else:
            cat_cols.append(c)
if drop_hi:
    print(f"dropping high-cardinality: {drop_hi}")

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

pre = ColumnTransformer([("num", num_pipe, num_cols),
                         ("cat", cat_pipe, cat_cols)],
                        remainder="drop")
try:
    pre.set_output(transform="pandas")
except Exception:
    pass

keep = num_cols + cat_cols
X_train = train[keep]
y_vec = y.values
fitted = mlca_runtime.fit_record(pre, X_train, "train", y=y_vec)
if fitted is None:
    fitted = pre

Xt_val = fitted.transform(val[keep])
print(f"val transformed shape: {Xt_val.shape}")

mlca_runtime.save_model(fitted, "preprocessor.pkl")
