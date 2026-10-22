import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
try:
    targets = list(proto["targets"])
    excluded = list(proto["excluded_columns"])
    task = proto["task"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available: {sorted(proto)}")
group = proto.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

exclude = set(targets) | set(excluded)
if group and group in train.columns:
    exclude.add(group)
feature_cols = [c for c in train.columns if c not in exclude]

def is_text(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)

key_like = [c for c in feature_cols if is_text(train[c]) and train[c].nunique(dropna=True) == len(train)]
for c in key_like:
    feature_cols.remove(c)

tokens = {"?", "-", "NA", "N/A"}
ph_cols = []
for c in list(feature_cols):
    if not is_text(train[c]):
        continue
    ss = train[c].astype("string").str.strip()
    bad = ss.isin(tokens) | (ss == "")
    if bad.any() or (train[c].astype("string") != ss).any():
        ph_cols.append(c)
        train[c] = ss.mask(bad, np.nan)
        if c in val.columns:
            vs = val[c].astype("string").str.strip()
            val[c] = vs.mask(vs.isin(tokens) | (vs == ""), np.nan)

date_cols = []
for c in list(feature_cols):
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif is_text(s):
        p = pd.to_datetime(s, errors="coerce")
        if p.notna().mean() > 0.9 and p.notna().sum() > 0:
            date_cols.append(c)
for c in date_cols:
    p = pd.to_datetime(train[c], errors="coerce")
    pv = pd.to_datetime(val[c], errors="coerce")
    parts = {
        "year": p.dt.year, "month": p.dt.month, "day": p.dt.day,
        "dayofweek": p.dt.dayofweek, "hour": p.dt.hour,
    }
    parts_v = {
        "year": pv.dt.year, "month": pv.dt.month, "day": pv.dt.day,
        "dayofweek": pv.dt.dayofweek, "hour": pv.dt.hour,
    }
    for k in parts:
        n = f"{c}_{k}"
        train[n] = parts[k]
        val[n] = parts_v[k]
        feature_cols.append(n)
    feature_cols.remove(c)

high = [c for c in feature_cols if is_text(train[c]) and train[c].nunique(dropna=True) > 0.05 * len(train)]
for c in high:
    feature_cols.remove(c)

num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_bool_dtype(train[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]

def screen_derived(df, cols, target):
    y = df[target]
    out = []
    for c in cols:
        s = df[c]
        same = (
            (pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(y))
            or (is_text(s) and is_text(y))
        )
        if same and s.astype("string").fillna("__NA__").equals(y.astype("string").fillna("__NA__")):
            out.append((c, "exact target copy"))
    return out

derived = []
if len(targets) == 1 and targets[0] in train.columns:
    derived = screen_derived(train, feature_cols, targets[0])
for c, _ in derived:
    if c in feature_cols:
        feature_cols.remove(c)
        if c in num_cols:
            num_cols.remove(c)
        if c in cat_cols:
            cat_cols.remove(c)

if len(targets) == 1 and pd.api.types.is_numeric_dtype(train[targets[0]]):
    nc = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
    if nc:
        cor = train[nc].corrwith(train[targets[0]]).abs()
        warn = cor[cor >= 0.98].index.tolist()
        if warn:
            print("Correlation >=0.98 kept: " + ", ".join(warn[:10]))

cat_width = sum(train[c].nunique(dropna=True) for c in cat_cols)
wide = len(num_cols) + cat_width > 300

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ("scaler", StandardScaler()),
])
transformers = [("num", num_pipe, num_cols)]
if cat_cols:
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=wide)),
    ])
    transformers.append(("cat", cat_pipe, cat_cols))

pre = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
if not wide:
    pre.set_output(transform="pandas")

X_train = train[feature_cols]
X_val = val[feature_cols]
y_train = train[targets[0]] if len(targets) == 1 else None
pre = mlca_runtime.fit_record(pre, X_train, "train", y=y_train)
Xt = pre.transform(X_val)

msg = "; ".join(f"{c} ({r})" for c, r in derived[:5]) + (" ..." if len(derived) > 5 else "")
print(f"Excluded derived features: {len(derived)}" + (f" [{msg}]" if derived else ""))
print(f"Placeholder tokens found in {len(ph_cols)} columns" + (": " + ", ".join(ph_cols[:5]) if ph_cols else ""))
print(f"Feature columns: {len(feature_cols)}; val transformed: {Xt.shape}")
mlca_runtime.save_model(pre, "preprocessor.pkl")
