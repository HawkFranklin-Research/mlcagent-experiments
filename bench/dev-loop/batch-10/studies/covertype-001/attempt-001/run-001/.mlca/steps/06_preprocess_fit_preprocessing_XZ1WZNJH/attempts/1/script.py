import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
target = targets[0]
n = len(train)

feat_cols = [c for c in train.columns
             if c not in targets and c not in excluded and c != group]

def key_like(s):
    if pd.api.types.is_float_dtype(s) or s.isna().any():
        return False
    return s.nunique() == len(s) and len(s) > 1

key_cols = [c for c in feat_cols if key_like(train[c])]
feat_cols = [c for c in feat_cols if c not in key_cols]
if key_cols:
    print(f"key-like dropped: {len(key_cols)}")

placeholders = {"?", "-", "NA", "N/A"}
ph_cols = []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        stripped = s.astype(str).str.strip()
        if stripped.isin(list(placeholders) + [""]).any():
            ph_cols.append(c)
if ph_cols:
    print(f"placeholder tokens in {len(ph_cols)} cols")
    for df in (train, val):
        for c in ph_cols:
            st = df[c].astype("string").str.strip()
            df[c] = st.replace(list(placeholders) + [""], pd.NA)
else:
    print("placeholder tokens: none")

target_arr = train[target].to_numpy()
nonneg = [c for c in feat_cols
          if pd.api.types.is_numeric_dtype(train[c]) and (train[c] >= 0).all()]
arrs = [(c, train[c].to_numpy()) for c in nonneg]
derived = []
for i in range(len(arrs)):
    c1, a1 = arrs[i]
    if np.array_equal(a1, target_arr):
        derived.append((c1, "equals target")); continue
    for j in range(i + 1, len(arrs)):
        c2, a2 = arrs[j]; s2 = a1 + a2
        if np.array_equal(s2, target_arr):
            derived.append((f"{c1}+{c2}", "sum equals target")); continue
        for k in range(j + 1, len(arrs)):
            c3, a3 = arrs[k]
            if np.array_equal(s2 + a3, target_arr):
                derived.append((f"{c1}+{c2}+{c3}", "sum equals target"))
                break
derived_cols = set()
for e, r in derived:
    print(f"excluded: {e} ({r})")
    derived_cols.update(e.split("+"))
feat_cols = [c for c in feat_cols if c not in derived_cols]

num_cols, cat_cols, hc = [], [], []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    elif s.nunique(dropna=True) > 0.05 * n:
        hc.append(c)
    else:
        cat_cols.append(c)
if hc:
    print(f"high-cardinality dropped: {len(hc)}")

parts = []
if num_cols:
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    num_pipe.set_output(transform="pandas")
    parts.append(("num", num_pipe, num_cols))
if cat_cols:
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    cat_pipe.set_output(transform="pandas")
    parts.append(("cat", cat_pipe, cat_cols))

pre = ColumnTransformer(parts, remainder="drop")
pre.set_output(transform="pandas")

X_train = train[feat_cols]
mlca_runtime.fit_record(pre, X_train, "train", y=train[target])
Xt = pre.transform(val[feat_cols])
print(f"train {X_train.shape} -> {Xt.shape[1]} encoded cols")

mlca_runtime.save_model(pre, "preprocessor.pkl")
mlca_runtime.record_metric("n_output_features", float(Xt.shape[1]), "train")
