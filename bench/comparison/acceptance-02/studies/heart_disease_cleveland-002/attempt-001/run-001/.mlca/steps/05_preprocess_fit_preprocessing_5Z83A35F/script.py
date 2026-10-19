import mlca_runtime, json, itertools
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pandas.api.types import (
    is_numeric_dtype, is_datetime64_any_dtype, is_object_dtype,
    is_string_dtype, is_bool_dtype,
)

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())
for key in ["targets", "excluded_columns", "task"]:
    if key not in proto:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(proto)}")
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group = proto.get("group_column")
if group and group not in excluded:
    excluded.append(group)

train = pd.read_parquet(paths["train"])
drop0 = set(targets) | set(excluded)
feature_cols = [c for c in train.columns if c not in drop0]

removed = []
for c in list(feature_cols):
    s = train[c]
    is_text = is_object_dtype(s) or is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)
    if is_text and s.notna().sum() > 1 and s.nunique(dropna=True) == s.notna().sum():
        feature_cols.remove(c)
        removed.append((c, "key-like text"))

X = train[feature_cols].copy()
derived = set()
for t in targets:
    if t not in train.columns:
        continue
    y = train[t]
    if not is_numeric_dtype(y) or not y.notna().all():
        continue
    num_candidates = [c for c in feature_cols if c not in derived and is_numeric_dtype(train[c])]
    for c in num_candidates:
        if train[c].notna().all() and (train[c] == y).all():
            derived.add(c)
            removed.append((c, f"equals target {t}"))
    for r in range(1, 4):
        for combo in itertools.combinations(num_candidates, r):
            if any(c in derived for c in combo):
                continue
            sub = train[list(combo)]
            if sub.isna().any().any() or (sub < 0).any().any():
                continue
            if (sub.sum(axis=1) == y).all():
                for c in combo:
                    derived.add(c)
                    removed.append((c, f"sum derivation of {t}"))
feature_cols = [c for c in feature_cols if c not in derived]
X = X[feature_cols].copy()

tokens = {"?", "-", "NA", "N/A"}
ph_cols = []
for c in feature_cols:
    s = X[c]
    if is_object_dtype(s) or is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        cleaned = s.astype("string").str.strip()
        bad = cleaned.isin(tokens) | (cleaned == "")
        if bad.any():
            ph_cols.append(c)
        X[c] = cleaned.mask(bad, pd.NA).astype(object)

new_cols = []
for c in list(feature_cols):
    s = X[c]
    dt = None
    if is_datetime64_any_dtype(s):
        dt = pd.to_datetime(s, errors="coerce")
    elif (is_object_dtype(s) or is_string_dtype(s)) and any(w in c.lower() for w in ("date", "time")):
        try:
            cand = pd.to_datetime(s, errors="coerce")
            if cand.notna().mean() > 0.9:
                dt = cand
        except Exception:
            pass
    if dt is not None:
        for part, val in [("year", dt.dt.year), ("month", dt.dt.month), ("day", dt.dt.day),
                          ("dayofweek", dt.dt.dayofweek), ("hour", dt.dt.hour)]:
            name = f"{c}_{part}"
            X[name] = val
            new_cols.append(name)
        feature_cols.remove(c)
        X.drop(columns=[c], inplace=True)
feature_cols.extend(new_cols)

n = len(X)
num_cols, cat_cols, high_card = [], [], []
for c in list(feature_cols):
    s = X[c]
    if is_numeric_dtype(s) and not is_bool_dtype(s):
        num_cols.append(c)
    elif is_object_dtype(s) or is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or is_bool_dtype(s):
        if s.nunique(dropna=True) > 0.05 * n:
            feature_cols.remove(c)
            X.drop(columns=[c], inplace=True)
            high_card.append(c)
        else:
            cat_cols.append(c)
    else:
        feature_cols.remove(c)
        X.drop(columns=[c], inplace=True)
        high_card.append(c)

warn = []
for t in targets:
    if t not in train.columns or not is_numeric_dtype(train[t]):
        continue
    for c in num_cols:
        d = pd.concat([train[c], train[t]], axis=1).dropna()
        if len(d) > 1 and d.iloc[:, 0].nunique() > 1 and d.iloc[:, 1].nunique() > 1:
            r = d.iloc[:, 0].corr(d.iloc[:, 1])
            if abs(r) >= 0.98:
                warn.append(c)

transformers = []
if num_cols:
    transformers.append((
        "num",
        Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
        num_cols,
    ))
width = len(num_cols) + sum(X[c].nunique(dropna=True) for c in cat_cols)
if cat_cols:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=width > 300)
    transformers.append((
        "cat",
        Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)]),
        cat_cols,
    ))
pre = ColumnTransformer(transformers=transformers, remainder="drop")
if width <= 300:
    pre.set_output(transform="pandas")

fitted = mlca_runtime.fit_record(pre, X[feature_cols], partition="train") or pre
mlca_runtime.save_model(fitted, "preprocessor.pkl")

print(f"features={len(feature_cols)} numeric={len(num_cols)} categorical={len(cat_cols)} width={width}")
print("placeholders: " + (f"{len(ph_cols)} cols" if ph_cols else "none"))
all_removed = removed + [(c, "high-cardinality") for c in high_card]
if all_removed:
    print("excluded: " + "; ".join(f"{c} ({r})" for c, r in all_removed[:5]) + (" ..." if len(all_removed) > 5 else ""))
if warn:
    print("corr>=0.98 warning: " + ", ".join(sorted(set(warn))))
print("saved preprocessor.pkl")
