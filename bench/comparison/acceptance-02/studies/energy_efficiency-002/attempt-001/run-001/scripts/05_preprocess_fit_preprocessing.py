import itertools
import json

import numpy as np
import pandas as pd

import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
if "targets" not in proto or "excluded_columns" not in proto or "task" not in proto:
    raise KeyError(f"protocol missing required keys; available: {list(proto)}")
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group = proto.get("group_column")
train = pd.read_parquet(inp["train"])
n = len(train)

skip = set(targets) | set(excluded) | ({group} if group else set())
feats = [c for c in train.columns if c not in skip]

placeholders = ["?", "-", "NA", "N/A", ""]
ph_cols = []
for c in feats:
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        st = s.astype("string").str.strip()
        if st.isin(placeholders).any():
            ph_cols.append(c)
            train[c] = st.mask(st.isin(placeholders), pd.NA).astype(object)

num_cols, cat_cols, dropped = [], [], {}
for c in feats:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        for part in ("year", "month", "day", "dayofweek"):
            name = f"{c}_{part}"
            train[name] = getattr(s.dt, part).astype(float)
            num_cols.append(name)
        dropped[c] = "datetime parsed into numeric parts"
    elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        if n > 1 and s.nunique(dropna=True) == n:
            dropped[c] = "key-like all-distinct integers"
        else:
            num_cols.append(c)
    else:
        u = s.nunique(dropna=True)
        if n > 1 and u == n:
            dropped[c] = "key-like all-distinct text codes"
        elif u > 0.05 * n:
            dropped[c] = "high-cardinality text (>5% unique)"
        else:
            cat_cols.append(c)

num_targets = [t for t in targets if pd.api.types.is_numeric_dtype(train[t])]
derived, nonneg = {}, [c for c in num_cols if train[c].notna().all() and (train[c] >= 0).all()]
for t in num_targets:
    yv = train[t].to_numpy()
    for c in list(num_cols):
        if c not in derived and train[c].notna().all() and np.array_equal(train[c].to_numpy(), yv):
            derived[c] = f"equals target {t}"
    for k in (1, 2, 3):
        for combo in itertools.combinations(nonneg, k):
            if any(c in derived for c in combo):
                continue
            if np.array_equal(train[list(combo)].sum(axis=1).to_numpy(), yv):
                for c in combo:
                    derived[c] = f"sum of up to 3 non-negative columns equals {t}"

corr_warn = []
for t in num_targets:
    for c in num_cols:
        if c in derived:
            continue
        v = train[c].astype(float).corr(train[t].astype(float))
        if v is not None and abs(v) >= 0.98:
            corr_warn.append(f"{c}~{t} r={v:.2f}")

for c in list(derived):
    dropped[c] = derived[c]
    if c in num_cols:
        num_cols.remove(c)
    if c in cat_cols:
        cat_cols.remove(c)

width = len(num_cols) + sum(int(train[c].nunique(dropna=True)) for c in cat_cols)
wide = width > 300
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=wide)),
])
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", cat_pipe, cat_cols),
], remainder="drop")
if not wide:
    pre.set_output(transform="pandas")

X = train[num_cols + cat_cols]
pre = mlca_runtime.fit_record(pre, X, "train")
mlca_runtime.save_model(pre, "preprocessor.pkl")

print(f"rows={n} numeric={len(num_cols)} categorical={len(cat_cols)} encoded_width={width}")
print(f"placeholder tokens found in {len(ph_cols)} columns: {ph_cols[:5]}")
print("excluded features: " + ("; ".join(f"{c} ({r})" for c, r in dropped.items()) if dropped else "none"))
print("correlation warnings (>=0.98, kept): " + (", ".join(corr_warn) if corr_warn else "none"))
