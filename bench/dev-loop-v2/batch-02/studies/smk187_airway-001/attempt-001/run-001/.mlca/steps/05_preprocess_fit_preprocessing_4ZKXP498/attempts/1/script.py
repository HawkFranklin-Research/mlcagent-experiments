import json
import numpy as np
import pandas as pd
import mlca_runtime
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_integer_dtype, is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
prot = json.loads(inp["_protocol"].read_text())

if "targets" not in prot:
    raise KeyError(f"missing 'targets'; available keys: {sorted(prot.keys())}")
targets = prot["targets"]
if not isinstance(targets, list):
    targets = [targets]
excluded = list(prot.get("excluded_columns", []) or [])
group = prot.get("group_column", None)
exclude_cols = set(targets) | set(excluded)
if group:
    exclude_cols.add(group)

candidates = [c for c in train.columns if c not in exclude_cols]
key_like = []
for c in candidates:
    s = train[c]
    if is_string_dtype(s) or s.dtype == object:
        if s.nunique(dropna=False) == len(train):
            key_like.append(c)
    elif is_integer_dtype(s) and s.nunique(dropna=False) == len(train):
        name = str(c).lower()
        if "id" in name or s.is_monotonic_increasing or s.is_monotonic_decreasing:
            key_like.append(c)
candidates = [c for c in candidates if c not in key_like]

for c in list(candidates):
    if is_datetime64_any_dtype(train[c]):
        for part in ["year", "month", "day", "dayofweek", "hour"]:
            name = f"{c}_{part}"
            train[name] = getattr(train[c].dt, part)
            val[name] = getattr(val[c].dt, part)
            candidates.append(name)
        candidates.remove(c)

y = train[targets[0]]
y_num = pd.to_numeric(y, errors="coerce")
derived = set()
derived_msgs = []
for c in candidates:
    if is_numeric_dtype(train[c]) and y_num.notna().all():
        eq = (train[c].to_numpy() == y_num.to_numpy()).all()
    else:
        eq = (train[c].astype("string").to_numpy() == y.astype("string").to_numpy()).all()
    if eq:
        derived.add(c)
        derived_msgs.append(f"{c}=={targets[0]}")

if y_num.notna().all():
    cand2 = [c for c in candidates if c not in derived and is_numeric_dtype(train[c])
             and train[c].notna().all() and (train[c] >= 0).all()
             and train[c].max() <= y_num.max()]
    arr = {c: train[c].to_numpy() for c in cand2}
    yv = y_num.to_numpy()
    seen = {}
    for c in cand2:
        h = hash((yv - arr[c]).tobytes())
        if h in seen:
            derived.update([c, seen[h]])
            derived_msgs.append(f"{c}+{seen[h]}=={targets[0]}")
        seen[hash(arr[c].tobytes())] = c
    if len(cand2) <= 60:
        pair = {}
        for i, c1 in enumerate(cand2):
            for c2 in cand2[i:]:
                pair[hash((arr[c1] + arr[c2]).tobytes())] = (c1, c2)
        for c3 in cand2:
            if hash((yv - arr[c3]).tobytes()) in pair:
                c1, c2 = pair[hash((yv - arr[c3]).tobytes())]
                derived.update([c1, c2, c3])
                derived_msgs.append(f"{c1}+{c2}+{c3}=={targets[0]}")

candidates = [c for c in candidates if c not in derived]
num_cols, cat_cols, dropped_high = [], [], []
for c in candidates:
    s = train[c]
    if is_numeric_dtype(s):
        num_cols.append(c)
    elif is_datetime64_any_dtype(s):
        continue
    else:
        nun = s.nunique(dropna=True)
        if nun > max(5, int(0.05 * len(train))):
            dropped_high.append(c)
        else:
            cat_cols.append(c)
feature_cols = num_cols + cat_cols

ph_cols, ph_tokens = 0, set()
for c in cat_cols:
    s = train[c].astype("string").str.strip()
    vs = val[c].astype("string").str.strip()
    mask = s.isin(["?", "-", "NA", "N/A"]) | s.eq("")
    vmask = vs.isin(["?", "-", "NA", "N/A"]) | vs.eq("")
    if mask.any():
        ph_cols += 1
        ph_tokens |= set(s[mask].dropna().unique())
        train[c] = s.mask(mask, pd.NA)
        val[c] = vs.mask(vmask, pd.NA)

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
if not transformers:
    raise ValueError("no feature columns available for preprocessing")
preprocessor = ColumnTransformer(transformers, remainder="drop")

est_width = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols) + len(cat_cols)
if est_width <= 300:
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")
    preprocessor.set_output(transform="pandas")

preprocessor = mlca_runtime.fit_record(preprocessor, X_train, "train")
X_val_trans = preprocessor.transform(X_val)

print(f"features={len(feature_cols)} numeric={len(num_cols)} categorical={len(cat_cols)} dropped_keylike={len(key_like)} dropped_highcard={len(dropped_high)}")
if derived_msgs:
    print("excluded derivations: " + "; ".join(derived_msgs[:3]))
if ph_tokens:
    print(f"placeholder tokens {sorted(ph_tokens)} in {ph_cols} categorical columns")
print(f"train_shape={X_train.shape} val_transformed_shape={X_val_trans.shape}")

mlca_runtime.record_metric("n_candidate_features", len(feature_cols), "train")
mlca_runtime.record_metric("n_transformed_features", int(X_val_trans.shape[1]), "val")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
