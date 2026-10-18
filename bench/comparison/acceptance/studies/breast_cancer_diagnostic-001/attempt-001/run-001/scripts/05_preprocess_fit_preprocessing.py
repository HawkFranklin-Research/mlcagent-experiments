import json
import itertools
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
protocol = json.loads(inputs["_protocol"].read_text())

def req(k):
    if k not in protocol:
        raise KeyError(f"Missing {k}; available top-level keys: {sorted(protocol)}")
    return protocol[k]

targets = list(req("targets"))
excluded = list(req("excluded_columns"))
group = req("group_column")
task = req("task")
target = targets[0]

drop = set(targets) | set(excluded)
if group is not None and not (isinstance(group, float) and pd.isna(group)):
    drop.add(group)

cand = [c for c in train.columns if c not in drop]

def is_keylike(s):
    if s.nunique(dropna=True) != len(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        return True
    return False

key_drop = [c for c in cand if is_keylike(train[c])]
feature_cols = [c for c in cand if c not in key_drop]
y = train[target]

derived = []
for c in feature_cols:
    try:
        if train[c].astype(object).equals(y.astype(object)):
            derived.append((c, "exact target"))
    except Exception:
        pass

if pd.api.types.is_numeric_dtype(y):
    num_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
    nonneg = [c for c in num_feats if train[c].min() >= 0]
    for r in (1, 2, 3):
        for comb in itertools.combinations(nonneg, r):
            if (train[list(comb)].sum(axis=1) == y).all():
                for c in comb:
                    if c not in {d[0] for d in derived}:
                        derived.append((c, "target sum"))

feature_cols = [c for c in feature_cols if c not in {d[0] for d in derived}]

cat_cols = [
    c for c in feature_cols
    if pd.api.types.is_object_dtype(train[c])
    or pd.api.types.is_string_dtype(train[c])
    or isinstance(train[c].dtype, pd.CategoricalDtype)
]
cat_low, cat_high = [], []
for c in cat_cols:
    (cat_high if train[c].nunique(dropna=True) > 0.05 * len(train) else cat_low).append(c)
feature_cols = [c for c in feature_cols if c not in cat_high]

num_cols = [c for c in feature_cols if c not in cat_cols]

ph_cols = []
for c in cat_cols:
    if c not in feature_cols:
        continue
    s = train[c].astype("string")
    stripped = s.str.strip()
    mask = stripped.isin(["?", "-", "NA", "N/A", ""]) | s.str.fullmatch(r"\s*")
    if mask.any():
        ph_cols.append(c)

corr_warn = []
if pd.api.types.is_numeric_dtype(y):
    for c in num_cols:
        try:
            if abs(train[c].corr(y)) >= 0.98:
                corr_warn.append(c)
        except Exception:
            pass

est_width = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_low)
use_pandas = est_width <= 300

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
parts = []
if num_cols:
    parts.append(("num", num_pipe, num_cols))
if cat_low:
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not use_pandas)),
    ])
    if use_pandas:
        cat_pipe.set_output(transform="pandas")
    parts.append(("cat", cat_pipe, cat_low))

ct = ColumnTransformer(parts, remainder="drop")
if use_pandas:
    ct.set_output(transform="pandas")

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()
ct = mlca_runtime.fit_record(ct, X_train, "train", y)
Xv = ct.transform(X_val)
mlca_runtime.save_model(ct, "preprocessor.pkl")
mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("val_transformed_width", float(Xv.shape[1]), "val")

print(f"Features: {len(feature_cols)}; key-like dropped: {len(key_drop)}; high-card cat dropped: {len(cat_high)}")
print("Excluded derived: " + (", ".join(f"{c} ({r})" for c, r in derived) if derived else "none"))
print("Correlation warnings: " + (", ".join(corr_warn) if corr_warn else ("none" if pd.api.types.is_numeric_dtype(y) else "skipped categorical target")))
print(f"Placeholder tokens in {len(ph_cols)} text columns")
print(f"Transformed val shape: {Xv.shape}")
