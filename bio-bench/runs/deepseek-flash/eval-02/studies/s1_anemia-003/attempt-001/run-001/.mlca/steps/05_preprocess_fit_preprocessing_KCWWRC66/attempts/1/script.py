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
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
protocol = json.loads(inp["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {list(d)}")
    return d[k]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
task = need(protocol, "task")
primary_metric = task.get("primary_metric")
group_col = protocol.get("group_column")
target = targets[0]

def clean_placeholders(df):
    found = {}
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            s = df[c].astype("string").str.strip()
            mask = s.isna() | s.isin(["?", "-", "NA", "N/A"]) | (s.str.len() == 0)
            if mask.any():
                found[c] = int(mask.sum())
                df[c] = s.mask(mask, np.nan)
            else:
                df[c] = s
    return df, found

train, ph_train = clean_placeholders(train)
val, _ = clean_placeholders(val)
print(f"Placeholder tokens found in {len(ph_train)} text columns" + (f": {list(ph_train)}" if ph_train else ""))

date_like = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    if pd.api.types.is_datetime64_any_dtype(train[c]):
        date_like.append(c)
    elif pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        sample = train[c].dropna().head(100)
        if len(sample):
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.9:
                date_like.append(c)

for c in date_like:
    for df in (train, val):
        dt = pd.to_datetime(df[c], errors="coerce")
        df[c + "_month"] = dt.dt.month
        df[c + "_dayofweek"] = dt.dt.dayofweek
        df[c + "_hour"] = dt.dt.hour
    train = train.drop(columns=[c])
    val = val.drop(columns=[c])

drop_set = set(targets) | set(excluded)
if group_col:
    drop_set.add(group_col)
cand = [c for c in train.columns if c not in drop_set]

def key_like(s):
    if s.isna().any() or s.nunique() != len(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        if set(s.unique()).issubset({0, 1}):
            return False
        return True
    return False

features = [c for c in cand if not key_like(train[c])]
y = train[target]
derived = set()

for c in features:
    if (train[c].fillna("__NA__").astype(str) == y.fillna("__NA__").astype(str)).all():
        derived.add(c)

if pd.api.types.is_numeric_dtype(y):
    num_feats = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    for r in (1, 2, 3):
        for combo in itertools.combinations(num_feats, r):
            if any((train[list(combo)] < 0).any()):
                continue
            if (train[list(combo)].sum(axis=1).round(10) == y.round(10)).all():
                derived.update(combo)

features = [c for c in features if c not in derived]
print(f"Excluded target-derived features: {sorted(derived) if derived else 'none'}")

warn = []
if pd.api.types.is_numeric_dtype(y):
    for c in features:
        if pd.api.types.is_numeric_dtype(train[c]):
            corr = train[c].corr(y)
            if abs(corr) >= 0.98:
                warn.append(c)
if warn:
    print(f"High target correlation kept: {warn}")

numeric = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
cat = [c for c in features if c not in numeric]
n = len(train)
cat = [c for c in cat if train[c].nunique(dropna=True) <= 0.05 * n]
features = numeric + cat

width = len(numeric) + sum(max(1, train[c].nunique(dropna=True)) for c in cat)
num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
pre = ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, cat)])

if width <= 300:
    pre.set_output(transform="pandas")
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")

mlca_runtime.fit_record(pre, train[features], "train", y=train[target])
val_t = pre.transform(val[features])
print(f"Fitted preprocessor on {train[features].shape}; val transformed {val_t.shape}")
mlca_runtime.save_model(pre, "preprocessor.pkl")
