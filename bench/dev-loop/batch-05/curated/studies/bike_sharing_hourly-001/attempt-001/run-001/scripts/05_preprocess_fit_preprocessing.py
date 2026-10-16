import json
import numpy as np
import pandas as pd
import mlca_runtime
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
proto = json.loads(inputs["_protocol"].read_text())

if "targets" not in proto or "excluded_columns" not in proto:
    raise KeyError(f"missing key; available: {sorted(proto.keys())}")

targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group_col = proto.get("group_column")
drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
cands = [c for c in train.columns if c not in drop]

def is_keylike(s):
    if s.isna().any() or s.nunique() != len(s):
        return False
    if pd.api.types.is_float_dtype(s):
        return False
    if pd.api.types.is_integer_dtype(s):
        v = np.sort(s.values)
        return bool(np.array_equal(v, np.arange(v[0], v[0] + len(v))))
    return True

key_like = [c for c in cands if is_keylike(train[c])]
features = [c for c in cands if c not in key_like]

y = train[targets[0]]
derived = []
if pd.api.types.is_numeric_dtype(y):
    yv = y.astype(float).values
    nn = [c for c in features if pd.api.types.is_numeric_dtype(train[c])
          and train[c].notna().all() and (train[c] >= 0).all()]
    for c in nn:
        if np.allclose(train[c].astype(float).values, yv):
            derived.append(c)
    for combo in list(combinations(nn, 2)) + list(combinations(nn, 3)):
        if np.allclose(sum(train[c].astype(float).values for c in combo), yv):
            derived += [c for c in combo if c not in derived]
            break
    for c in derived:
        print(f"excluded derivation: {c}")
    features = [c for c in features if c not in derived]
    for c in features:
        if pd.api.types.is_numeric_dtype(train[c]) and train[c].notna().all():
            r = abs(np.corrcoef(train[c].astype(float).values, yv)[0, 1])
            if r >= 0.98:
                print(f"corr_warn: {c} r={r:.3f}")

obj_cols = [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]
ph_hits, ph_cols = 0, 0
for c in obj_cols:
    s = train[c].astype(str)
    n = int(s.str.strip().isin(["?", "-", "NA", "N/A"]).sum() + (s.str.strip() == "").sum())
    if n:
        ph_hits += n
        ph_cols += 1
if ph_hits:
    print(f"placeholders: {ph_hits} in {ph_cols} cols")

num_cols = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in features if c not in num_cols]
thr = max(1, int(0.05 * len(train)))
low_card = [c for c in cat_cols if train[c].nunique(dropna=True) <= thr]
high_card = [c for c in cat_cols if c not in low_card]
for c in high_card:
    print(f"drop_high_card: {c} uniq={train[c].nunique(dropna=True)}")

keep = num_cols + low_card
num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
ct = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, low_card)],
                       remainder="drop")
ct.set_output(transform="pandas")

X = train[keep]
mlca_runtime.fit_record(ct, X, "train", y=y.values)
mlca_runtime.save_model(ct, "preprocessor.pkl")
print(f"fit: features={len(keep)} numeric={len(num_cols)} low_card={len(low_card)} targets={targets}")
