import json
import itertools
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    group = protocol["group_column"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e.args[0]!r}; available keys: {sorted(protocol)}")

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
cands = [c for c in train.columns if c not in drop]

text_cols = [
    c for c in cands
    if pd.api.types.is_object_dtype(train[c])
    or pd.api.types.is_string_dtype(train[c])
    or isinstance(train[c].dtype, pd.CategoricalDtype)
]
placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = []
for c in text_cols:
    s = train[c].astype("string")
    stripped = s.str.strip()
    has_ph = stripped.isin(placeholders) | (s.notna() & (s != stripped))
    if has_ph.any():
        ph_cols.append(c)
        train[c] = stripped.mask(stripped.isin(placeholders), np.nan).astype(object)

features = []
for c in cands:
    s = train[c]
    distinct = s.nunique(dropna=True)
    nonnull = int(s.notna().sum())
    key_like = (
        (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
         or isinstance(s.dtype, pd.CategoricalDtype)
         or pd.api.types.is_integer_dtype(s))
        and nonnull == len(s) and distinct == nonnull
    )
    if not key_like:
        features.append(c)

y = train[targets[0]]
mask = y.notna()
deriv = {}
target_numeric = pd.api.types.is_numeric_dtype(y)

for c in features:
    s = train[c]
    if mask.any() and np.array_equal(s[mask].to_numpy(), y[mask].to_numpy()):
        deriv[c] = "equals target"

if target_numeric and mask.any():
    num_cands = [
        c for c in features
        if pd.api.types.is_numeric_dtype(train[c])
        and (train[c].dropna() >= 0).all()
    ]
    for r in (2, 3):
        for combo in itertools.combinations(num_cands, r):
            vals = train.loc[mask, list(combo)].sum(axis=1)
            if np.allclose(vals.to_numpy(), y[mask].to_numpy(), rtol=0, atol=1e-12):
                for c in combo:
                    deriv[c] = "target sum"

features = [c for c in features if c not in deriv]
num_cols = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in features if c not in num_cols]
threshold = max(1, int(0.05 * len(train)))
cat_cols = [c for c in cat_cols if train[c].nunique(dropna=True) <= threshold]

print(f"train rows={len(train)} candidate features={len(cands)}")
print(f"placeholders={len(ph_cols)} columns" if ph_cols else "placeholders: none")
print("excluded derivations:", "; ".join(f"{c}:{r}" for c, r in deriv.items()) if deriv else "none")
if target_numeric:
    warn = []
    for c in num_cols:
        corr = train[c].corr(y)
        if pd.notna(corr) and abs(corr) >= 0.98:
            warn.append(c)
    if warn:
        print("corr>=0.98:", ", ".join(warn))

est_width = len(num_cols) + int(sum(train[c].nunique(dropna=True) for c in cat_cols))
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=est_width > 300)
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", onehot),
])
transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))

pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
if est_width <= 300:
    pre.set_output(transform="pandas")

X_train = train[features].copy()
mlca_runtime.fit_record(pre, X_train, "train")
mlca_runtime.save_model(pre, "preprocessor.pkl")
print(f"preprocessor numeric={len(num_cols)} categorical={len(cat_cols)}")
