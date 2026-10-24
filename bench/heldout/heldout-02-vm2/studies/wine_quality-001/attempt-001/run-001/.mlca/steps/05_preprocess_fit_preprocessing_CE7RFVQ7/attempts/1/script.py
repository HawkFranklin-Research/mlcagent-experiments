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
val = pd.read_parquet(inp["val"])
protocol = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in protocol:
        raise KeyError(f"Missing {k}; available top-level keys: {list(protocol)}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
task = protocol["task"]
group = protocol.get("group_column", None)

def is_str(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)

def is_date_like(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not is_str(s):
        return False
    v = s.dropna().astype(str).head(100)
    if len(v) == 0:
        return False
    return v.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}([ T].*)?$").mean() > 0.9

base = [c for c in train.columns if c not in set(targets + excluded) and c != group]
date_cols = [c for c in base if is_date_like(train[c])]
for c in date_cols:
    dt, dv = pd.to_datetime(train[c], errors="coerce"), pd.to_datetime(val[c], errors="coerce")
    for suf in ("year", "month", "day", "dayofweek", "hour"):
        train[f"{c}_{suf}"] = getattr(dt.dt, suf)
        val[f"{c}_{suf}"] = getattr(dv.dt, suf)
        base.append(f"{c}_{suf}")
    base.remove(c)

key_like = []
for c in base:
    s = train[c]
    n = s.notna().sum()
    if n and s.nunique(dropna=True) == n and not is_date_like(s):
        if is_str(s) or pd.api.types.is_integer_dtype(s):
            key_like.append(c)
base = [c for c in base if c not in key_like]

high_card = []
for c in base:
    s = train[c]
    if (is_str(s) or isinstance(s.dtype, pd.CategoricalDtype)) and not pd.api.types.is_numeric_dtype(s):
        if s.nunique(dropna=True) > 0.05 * len(train):
            high_card.append(c)
base = [c for c in base if c not in high_card]

derived = []
for t in targets:
    y = train[t]
    for c in list(base):
        if c not in derived and train[c].eq(y).all():
            derived.append(c)
    base2 = [c for c in base if c not in derived]
    if pd.api.types.is_numeric_dtype(y):
        nonneg = [c for c in base2 if pd.api.types.is_numeric_dtype(train[c]) and train[c].min(skipna=True) >= 0]
        found = True
        while found:
            found = False
            for r in (2, 3):
                for combo in itertools.combinations(nonneg, r):
                    if any(c in derived for c in combo):
                        continue
                    if train[list(combo)].sum(axis=1).eq(y).all():
                        derived.extend(combo)
                        found = True
                        break
                if found:
                    break
            nonneg = [c for c in nonneg if c not in derived]
base = [c for c in base if c not in derived]

feature_cols = base
num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]

ph_tokens = ["?", "-", "NA", "N/A"]
ph_cols = []
for c in feature_cols:
    s = train[c]
    if is_str(s) or isinstance(s.dtype, pd.CategoricalDtype):
        raw = s.dropna().astype(str)
        stripped = raw.str.strip()
        if stripped.isin(ph_tokens + [""]).any() or (raw != stripped).any():
            ph_cols.append(c)

n_out = len(num_cols) + sum(max(1, train[c].nunique(dropna=True)) for c in cat_cols)
wide = n_out > 300
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=not wide)
num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)])
transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
ct = ColumnTransformer(transformers, remainder="drop")
if not wide:
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")
    ct.set_output(transform="pandas")

pre = ct
if ph_cols:
    pre = Pipeline([
        ("strip", FunctionTransformer(func=pd.DataFrame.replace, kw_args={"to_replace": r"^\s+|\s+$", "value": "", "regex": True})),
        ("missing", FunctionTransformer(func=pd.DataFrame.replace, kw_args={"to_replace": ph_tokens + [""], "value": np.nan})),
        ("ct", ct),
    ])
    if not wide:
        pre.set_output(transform="pandas")

X_train = train[feature_cols]
X_val = val[feature_cols]
mlca_runtime.fit_record(pre, X_train, "train")
Xt = pre.transform(X_val)
mlca_runtime.save_model(pre, "preprocessor.pkl")
mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_transformed_cols", float(Xt.shape[1]), "val")

print(f"Features: {len(feature_cols)} (num={len(num_cols)}, cat={len(cat_cols)})")
print(f"Target-derived excluded: {len(derived)}")
print(f"Dropped key-like/high-card: {len(key_like) + len(high_card)}")
print(f"Placeholder columns: {len(ph_cols)}")
print(f"Val transformed: {Xt.shape}")
