import json, itertools
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())

def req(k):
    if k not in proto:
        raise KeyError(f"Missing protocol key {k}; available: {list(proto)}")
    return proto[k]

targets = req("targets")
excluded = req("excluded_columns")
group = req("group_column")
task = req("task")

df = pd.read_parquet(inp["train"])
excl_cols = set(targets) | set(excluded) | ({group} if group else set())
base = [c for c in df.columns if c not in excl_cols]

def key_like(s):
    if s.isna().any() or s.nunique(dropna=True) != len(s):
        return False
    if pd.api.types.is_integer_dtype(s):
        u = np.sort(s.unique())
        return len(u) > 1 and np.array_equal(u, np.arange(u.min(), u.min() + len(u)))
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)

feats = [c for c in base if not key_like(df[c])]

def placeholder_clean(frame, cols):
    tokens = {"?", "-", "NA", "N/A"}
    found = []
    for c in cols:
        if pd.api.types.is_object_dtype(frame[c]) or pd.api.types.is_string_dtype(frame[c]):
            s = frame[c].astype("string").str.strip()
            mask = s.isin(tokens) | s.eq("")
            if mask.any():
                found.append(c)
                frame[c] = s.mask(mask, pd.NA)
    return found

ph = placeholder_clean(df, feats)
if ph:
    print(f"placeholders cleaned in {len(ph)} text columns")

def derivations(frame, cols, target):
    out = {}
    for c in cols:
        if frame[c].notna().all() and (frame[c] == frame[target]).all():
            out[c] = "equals target"
    nonneg = [c for c in cols if c not in out and pd.api.types.is_numeric_dtype(frame[c]) and frame[c].dropna().ge(0).all()]
    for r in range(1, min(3, len(nonneg)) + 1):
        for combo in itertools.combinations(nonneg, r):
            if any(c in out for c in combo):
                continue
            s = frame[list(combo)].sum(axis=1)
            if s.notna().all() and (s == frame[target]).all():
                for c in combo:
                    out[c] = "sum equals target"
    return out

der = derivations(df, feats, targets[0])
if der:
    print("excluded derivations: " + "; ".join(f"{c} ({why})" for c, why in der.items()))
feats = [c for c in feats if c not in der]

if pd.api.types.is_numeric_dtype(df[targets[0]]):
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    corr = df[num].corrwith(df[targets[0]]).abs()
    warn = corr[corr >= 0.98].index.tolist()
    if warn:
        print("corr>=0.98 kept:", ", ".join(warn))

dtc = []
for c in feats:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dtc.append(c)
    elif (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and pd.api.types.infer_dtype(s, skipna=True) == "datetime":
        dtc.append(c)
for c in dtc:
    p = pd.to_datetime(df[c], errors="coerce")
    for part in ["year", "month", "day", "dayofweek", "hour"]:
        df[f"{c}_{part}"] = getattr(p.dt, part)
        feats.append(f"{c}_{part}")
    feats.remove(c)

cat = [c for c in feats if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)]
high = [c for c in cat if df[c].nunique(dropna=True) > max(5, 0.05 * len(df))]
if high:
    print(f"dropped high-cardinality: {len(high)}")
feats = [c for c in feats if c not in high]
cat = [c for c in cat if c not in high]
num = [c for c in feats if c not in cat]

n_out = len(num) + sum(df[c].nunique(dropna=True) for c in cat)
pandas_out = n_out <= 300
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not pandas_out))])
trans = []
if num: trans.append(("num", num_pipe, num))
if cat: trans.append(("cat", cat_pipe, cat))
pre = ColumnTransformer(trans, remainder="drop", verbose_feature_names_out=False)
if pandas_out:
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")
    pre.set_output(transform="pandas")

X = df[feats].copy()
mlca_runtime.fit_record(pre, X, "train")
mlca_runtime.save_model(pre, "preprocessor.pkl")
mlca_runtime.record_metric("n_features", float(len(feats)), "train")
print(f"features: {len(feats)} ({len(num)} numeric, {len(cat)} categorical), output cols ~{n_out}")
