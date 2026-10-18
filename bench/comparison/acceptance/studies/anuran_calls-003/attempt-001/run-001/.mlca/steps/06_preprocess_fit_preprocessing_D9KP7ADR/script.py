import mlca_runtime
import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
P = json.loads(inp["_protocol"].read_text())
try:
    targets = P["targets"]
    excluded = P["excluded_columns"]
    group = P["group_column"]
except KeyError as e:
    raise KeyError(f"Missing {e.args[0]!r}; available keys: {sorted(P)}")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

candidate = [c for c in train.columns if c not in set(targets + excluded + [group])]
key_like = []
for c in candidate:
    s = train[c]
    if s.nunique(dropna=False) == len(train) and (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_integer_dtype(s)
    ):
        key_like.append(c)
candidate = [c for c in candidate if c not in key_like]

ph_cols = 0
ph_vals = set()
for c in [c for c in candidate if not pd.api.types.is_numeric_dtype(train[c])]:
    s = train[c].astype("string")
    stripped = s.str.strip()
    mask = stripped.isin(["?", "-", "NA", "N/A"]) | (s.notna() & (stripped == ""))
    if mask.any():
        ph_cols += 1
        ph_vals.update(stripped[mask].dropna().unique().tolist())
        train[c] = stripped.mask(stripped.isin(["?", "-", "NA", "N/A"]) | (stripped == ""), pd.NA)

def _eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":
        return bool((pd.isna(a) == pd.isna(b)).all() and (a[pd.notna(a)] == b[pd.notna(a)]).all())
    return bool(np.array_equal(a, b, equal_nan=True))

excluded_feats = {}
for t in targets:
    for c in candidate:
        if c not in excluded_feats and _eq(train[c].to_numpy(), train[t].to_numpy()):
            excluded_feats[c] = f"exact equality with {t}"
candidate = [c for c in candidate if c not in excluded_feats]

for t in targets:
    if not (pd.api.types.is_numeric_dtype(train[t]) and not pd.api.types.is_bool_dtype(train[t])):
        continue
    y = train[t].to_numpy()
    if pd.isna(y).any():
        continue
    ncols = [
        c for c in candidate
        if c not in excluded_feats
        and pd.api.types.is_numeric_dtype(train[c])
        and not pd.api.types.is_bool_dtype(train[c])
        and train[c].notna().all()
        and (train[c] >= 0).all()
    ]
    found = set()
    for r in (1, 2, 3):
        for combo in combinations(ncols, r):
            if found.intersection(combo):
                continue
            s = train[list(combo)].sum(axis=1).to_numpy()
            if np.allclose(s, y, rtol=0, atol=1e-12):
                for c in combo:
                    excluded_feats[c] = f"row sum equals {t}"
                found.update(combo)
    for c in found:
        if c in candidate:
            candidate.remove(c)

warn = []
for t in targets:
    if pd.api.types.is_numeric_dtype(train[t]) and not pd.api.types.is_bool_dtype(train[t]):
        for c in candidate:
            if pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c]):
                corr = train[[c, t]].corr().iloc[0, 1]
                if pd.notna(corr) and abs(corr) >= 0.98:
                    warn.append(c)

feature_cols = list(candidate)
num_cols, cat_cols, high_card = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    elif s.nunique(dropna=False) <= max(1, int(0.05 * len(train))):
        cat_cols.append(c)
    else:
        high_card.append(c)
feature_cols = [c for c in feature_cols if c not in high_card]
num_cols = [c for c in num_cols if c in feature_cols]
cat_cols = [c for c in cat_cols if c in feature_cols]
if not feature_cols:
    raise ValueError("No usable feature columns after screening")

cat_width = sum(train[c].nunique(dropna=False) for c in cat_cols)
wide = len(num_cols) + cat_width > 300

num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not wide)),
])
transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
if not wide:
    pre.set_output(transform="pandas")
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")

pre = mlca_runtime.fit_record(pre, train[feature_cols], "train")
pre.feature_columns_ = list(feature_cols)
mlca_runtime.save_model(pre, "preprocessor.pkl")
mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_train_rows", float(len(train)), "train")

if ph_cols:
    print(f"placeholders: {ph_cols} columns {sorted(ph_vals)}")
else:
    print("placeholders: none")
print("corr_warn: " + ",".join(sorted(set(warn))) if warn else "corr_warn: none")
if high_card:
    print("dropped_high_card: " + ",".join(high_card))
print(f"fit: rows={len(train)} features={len(feature_cols)} num={len(num_cols)} cat={len(cat_cols)} wide={wide}")
print("excluded: " + ("; ".join(f"{k} ({v})" for k, v in excluded_feats.items()) if excluded_feats else "none"))
