import json, itertools
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pandas.api.types import (
    is_numeric_dtype, is_bool_dtype, is_object_dtype, is_string_dtype,
    is_integer_dtype,
)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {sorted(d.keys())}")
    return d[k]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
task = need(protocol, "task")
group = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
for c in targets + excluded:
    if c not in train.columns:
        raise KeyError(f"Missing declared column {c!r} in train")

base_cols = [c for c in train.columns if c not in set(targets + excluded + ([group] if group else []))]
placeholders = {"?", "-", "NA", "N/A", ""}
ph_tokens, ph_cols = set(), 0
for c in base_cols:
    if c not in train.columns or c not in val.columns:
        continue
    if is_object_dtype(train[c]) or is_string_dtype(train[c]):
        found = False
        for df in (train, val):
            s = df[c].astype("string")
            st = s.str.strip()
            m = st.isin(placeholders)
            if m.any():
                found = True
                ph_tokens.update(st[m].dropna().unique())
            df[c] = st.mask(m, pd.NA)
        ph_cols += int(found)

for c in base_cols:
    if c not in train.columns:
        continue
    if is_object_dtype(train[c]) or is_string_dtype(train[c]):
        conv = pd.to_numeric(train[c], errors="coerce")
        nonnull = int(train[c].notna().sum())
        if nonnull and int(conv.notna().sum()) == nonnull:
            train[c] = conv
            if c in val.columns:
                val[c] = pd.to_numeric(val[c], errors="coerce")

n = len(train)
feature_cols, drop_info = [], []
for c in base_cols:
    if c not in train.columns:
        continue
    s = train[c]
    nn, nu = int(s.notna().sum()), int(s.nunique(dropna=True))
    if is_numeric_dtype(s):
        if is_integer_dtype(s) and nn == n and nu == n:
            drop_info.append((c, "integer key-like"))
            continue
    else:
        if nn == n and nu == n:
            drop_info.append((c, "text key-like"))
            continue
        if nu > max(1, 0.05 * n):
            drop_info.append((c, "high-cardinality"))
            continue
    feature_cols.append(c)

y = train[target]
excl, corr_warn = [], []
if is_numeric_dtype(y):
    ynum = pd.to_numeric(y, errors="coerce")
    for c in list(feature_cols):
        if is_numeric_dtype(train[c]):
            xnum = pd.to_numeric(train[c], errors="coerce")
            if xnum.notna().all() and ynum.notna().all() and (xnum.values == ynum.values).all():
                feature_cols.remove(c)
                excl.append((c, "equals target"))
    numcols = [c for c in feature_cols if is_numeric_dtype(train[c]) and train[c].notna().all() and train[c].min() >= 0]
    for r in (1, 2, 3):
        for combo in itertools.combinations(numcols, r):
            if any(c in [x[0] for x in excl] for c in combo):
                continue
            sm = train[list(combo)].sum(axis=1)
            if sm.notna().all() and ynum.notna().all() and (sm.values == ynum.values).all():
                for c in combo:
                    if c in feature_cols:
                        feature_cols.remove(c)
                        excl.append((c, f"sum-derivation ({','.join(combo)})"))
    for c in feature_cols:
        if is_numeric_dtype(train[c]):
            rho = train[c].corr(ynum)
            if pd.notna(rho) and abs(rho) >= 0.98:
                corr_warn.append(c)
else:
    ys = y.astype("string")
    for c in list(feature_cols):
        if train[c].notna().all() and y.notna().all() and train[c].astype("string").eq(ys).all():
            feature_cols.remove(c)
            excl.append((c, "equals target"))

numeric_cols = [c for c in feature_cols if is_numeric_dtype(train[c]) or is_bool_dtype(train[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]
width = len(numeric_cols) + sum(int(train[c].nunique(dropna=False)) for c in categorical_cols)

transformers = []
if numeric_cols:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    transformers.append(("num", num_pipe, numeric_cols))
if categorical_cols:
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=width > 300)),
    ])
    transformers.append(("cat", cat_pipe, categorical_cols))

pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
if width <= 300:
    for _, trans, _ in transformers:
        try:
            trans.set_output(transform="pandas")
        except Exception:
            pass
    try:
        pre.set_output(transform="pandas")
    except Exception:
        pass

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()
fitted = mlca_runtime.fit_record(pre, X_train, "train", y=train[target])
Xt_val = fitted.transform(X_val)
mlca_runtime.save_model(fitted, "preprocessor.pkl")

fmt = lambda items: "none" if not items else "; ".join(f"{a}:{b}" for a, b in items)
print(f"placeholders: {ph_cols} cols tokens={','.join(sorted(x if x else '<whitespace>' for x in ph_tokens)) or 'none'}")
print(f"excluded features: {fmt(drop_info + excl)}")
if corr_warn:
    print("corr>=0.98 kept: " + ",".join(corr_warn))
print(f"features={len(feature_cols)} numeric={len(numeric_cols)} categorical={len(categorical_cols)} width~={width}")
print(f"val transformed shape={getattr(Xt_val, 'shape', None)}")
