import mlca_runtime
import json
import itertools
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
protocol = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns"):
    if k not in protocol:
        raise KeyError(f"Missing {k}; available: {sorted(protocol)}")
targets = protocol.get("targets", [])
excluded = set(protocol.get("excluded_columns", []))
group_col = protocol.get("group_column")
drop = excluded | set(targets)
if group_col:
    drop.add(group_col)
feature_cols = [c for c in train.columns if c not in drop]

def keylike(s):
    if s.dtype.kind in "OU" or str(s.dtype) in ("string", "category"):
        return len(s) > 0 and s.nunique(dropna=True) == len(s)
    if s.dtype.kind in "iu":
        return len(s) > 0 and s.nunique(dropna=True) == len(s)
    return False

feature_cols = [c for c in feature_cols if not keylike(train[c])]
X = train[feature_cols].copy()
placeholders = {"?", "-", "NA", "N/A"}
text_cols = [c for c in X.columns if X[c].dtype.kind in "OU" or str(X[c].dtype) in ("string", "category")]
placeholder_cols = []
for c in text_cols:
    s = X[c].astype("string").str.strip()
    t = s.fillna("__NA__")
    mask = t.isin(placeholders) | t.eq("")
    if mask.any():
        placeholder_cols.append(c)
    X[c] = s.mask(mask, np.nan)

derived = []
if targets:
    for t in targets:
        if t not in train:
            continue
        tv = train[t]
        for c in list(X.columns):
            if c not in X or c in [d[0] for d in derived]:
                continue
            if X[c].eq(tv).all():
                derived.append((c, f"equals {t}"))
        nonneg = [c for c in X.columns if c not in [d[0] for d in derived]
                  and pd.api.types.is_numeric_dtype(X[c]) and X[c].min(skipna=True) >= 0]
        for r in (1, 2, 3):
            for combo in itertools.combinations(nonneg, r):
                if X[list(combo)].sum(axis=1).eq(tv).all():
                    for c in combo:
                        if c not in [d[0] for d in derived]:
                            derived.append((c, f"sums to {t}"))
                    break
if derived:
    X = X.drop(columns=[c for c, _ in derived])

warn = []
if targets:
    for t in targets:
        if t in train and pd.api.types.is_numeric_dtype(train[t]):
            for c in X.columns:
                if pd.api.types.is_numeric_dtype(X[c]):
                    corr = X[c].astype(float).corr(train[t].astype(float))
                    if abs(corr) >= 0.98:
                        warn.append(c)

date_cols = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]
for c in date_cols:
    dt = pd.to_datetime(X[c])
    X[c + "_hour"] = dt.dt.hour
    X[c + "_dow"] = dt.dt.dayofweek
    X[c + "_month"] = dt.dt.month
X = X.drop(columns=date_cols)

cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
high = [c for c in cat_cols if X[c].nunique(dropna=True) > 0.05 * len(X)]
X = X.drop(columns=high)
cat_cols = [c for c in cat_cols if c not in high and X[c].notna().any()]
num_cols = [c for c in X.columns if c not in cat_cols and X[c].notna().any()]
X = X[num_cols + cat_cols]
log_cols = [c for c in num_cols if X[c].min(skipna=True) >= 0 and abs(X[c].skew()) > 1]
plain_num = [c for c in num_cols if c not in log_cols]

transformers = []
if plain_num:
    p = Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    p.set_output(transform="pandas")
    transformers.append(("num", p, plain_num))
if log_cols:
    p = Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                  ("scale", StandardScaler())])
    p.set_output(transform="pandas")
    transformers.append(("log", p, log_cols))
if cat_cols:
    width_guess = len(plain_num) + len(log_cols) + sum(int(X[c].nunique(dropna=True)) for c in cat_cols)
    p = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                  ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=width_guess > 300))])
    p.set_output(transform="pandas")
    transformers.append(("cat", p, cat_cols))

if not transformers:
    raise ValueError("No usable feature columns after preprocessing selection")
ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
if not (len(plain_num) + len(log_cols) + sum(int(X[c].nunique(dropna=True)) for c in cat_cols) > 300):
    ct.set_output(transform="pandas")
ct = mlca_runtime.fit_record(ct, X, "train")
mlca_runtime.save_model(ct, "preprocessor.pkl")
width = len(ct.get_feature_names_out())
mlca_runtime.record_metric("n_features", float(X.shape[1]), "train")
mlca_runtime.record_metric("placeholder_columns", float(len(placeholder_cols)), "train")
mlca_runtime.record_metric("excluded_derivations", float(len(derived)), "train")
mlca_runtime.record_metric("output_width", float(width), "train")
print(f"features {X.shape[1]} rows {len(X)}; numeric {len(num_cols)} categorical {len(cat_cols)} highcard_dropped {len(high)}")
print("placeholders: " + (", ".join(placeholder_cols[:5]) if placeholder_cols else "none"))
print("derived excluded: " + (", ".join(f"{c}({r})" for c, r in derived[:5]) if derived else "none"))
if warn:
    print("corr>=0.98 kept: " + ", ".join(warn[:10]))
print(f"saved preprocessor.pkl width {width}")
