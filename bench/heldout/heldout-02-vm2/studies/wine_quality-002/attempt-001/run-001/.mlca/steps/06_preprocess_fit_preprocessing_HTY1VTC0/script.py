import json
import itertools
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
protocol = json.loads(inp["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {list(d.keys())}")
    return d[k]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group = need(protocol, "group_column")
task = need(protocol, "task")
target_col = targets[0]

exclude = set(targets) | set(excluded) | ({group} if group else set())
candidate = [c for c in train.columns if c not in exclude]

# placeholder tokens in text columns
placeholders = {"?", "-", "NA", "N/A", ""}
placeholder_cols = []
for c in list(candidate):
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        s = train[c].astype("string").str.strip()
        mask = s.isin(placeholders) | s.eq("")
        if mask.any():
            placeholder_cols.append(c)
            train[c] = s.mask(mask, np.nan).astype(object)
            vs = val[c].astype("string").str.strip()
            vm = vs.isin(placeholders) | vs.eq("")
            val[c] = vs.mask(vm, np.nan).astype(object)

# date handling: derive numeric parts then drop raw datetime/date strings
dropped_dates = set()
for c in list(candidate):
    is_dt = pd.api.types.is_datetime64_any_dtype(train[c])
    name_dt = any(k in c.lower() for k in ("date", "time", "_dt"))
    if is_dt or (name_dt and (pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]))):
        dt = pd.to_datetime(train[c], errors="coerce")
        vdt = pd.to_datetime(val[c], errors="coerce")
        if dt.notna().mean() > 0.9:
            for part, tr, va in [
                ("year", dt.dt.year, vdt.dt.year),
                ("month", dt.dt.month, vdt.dt.month),
                ("day", dt.dt.day, vdt.dt.day),
                ("dow", dt.dt.dayofweek, vdt.dt.dayofweek),
            ]:
                new = f"{c}_{part}"
                train[new] = tr
                val[new] = va
                candidate.append(new)
            dropped_dates.add(c)
candidate = [c for c in candidate if c not in dropped_dates]

# key-like columns: all-distinct text codes or integer sequences
key_like = []
for c in list(candidate):
    s = train[c]
    n = len(train)
    distinct = s.nunique(dropna=True)
    if distinct == n and not pd.api.types.is_bool_dtype(s):
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            key_like.append(c)
        elif pd.api.types.is_integer_dtype(s) and not set(s.dropna().unique()).issubset({0, 1}):
            key_like.append(c)
candidate = [c for c in candidate if c not in key_like]

# exact-derivation screen on training rows
num_cand = [c for c in candidate if pd.api.types.is_numeric_dtype(train[c])]
t = pd.to_numeric(train[target_col], errors="coerce")
derived, reason = set(), {}
for c in num_cand:
    if train[c].notna().all() and (train[c] == t).all():
        derived.add(c)
        reason[c] = "equals target"
nonneg = [c for c in num_cand if train[c].notna().all() and (train[c] >= 0).all()]
for r in (1, 2, 3):
    for combo in itertools.combinations(nonneg, r):
        s = train[list(combo)].sum(axis=1)
        if (s == t).all():
            for c in combo:
                derived.add(c)
                reason[c] = f"row sum of {r} non-negative columns equals target"
candidate = [c for c in candidate if c not in derived]

# type split
numeric_cols = [c for c in candidate if pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_bool_dtype(train[c])]
cat_cols = [c for c in candidate if c not in numeric_cols]
max_cat = max(1, int(0.05 * len(train)))
cat_cols = [c for c in cat_cols if train[c].nunique(dropna=True) <= max_cat]

feature_cols = numeric_cols + cat_cols
width = len(numeric_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
sparse = width > 300

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)),
])
transformers = []
if numeric_cols:
    transformers.append(("num", num_pipe, numeric_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))

preprocessor = ColumnTransformer(transformers, remainder="drop")
if not sparse:
    preprocessor.set_output(transform="pandas")

X_train = train[feature_cols]
X_val = val[feature_cols]
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, "train")
val_t = preprocessor.transform(X_val)

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("encoded_width", float(width), "train")

print(f"features: {len(feature_cols)} numeric={len(numeric_cols)} categorical={len(cat_cols)}")
print(f"encoded width: {width} ({'sparse' if sparse else 'pandas'})")
if placeholder_cols:
    print(f"placeholders found in {len(placeholder_cols)} columns: {placeholder_cols}")
if derived:
    print("derived features excluded: " + ", ".join(f"{c} ({reason[c]})" for c in sorted(derived)))
print(f"val transformed shape: {val_t.shape}")
