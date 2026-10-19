import mlca_runtime, json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])

def need(d, key):
    if key not in d:
        raise KeyError(f"missing {key}; available top-level keys: {list(d)}")
    return d[key]

targets = need(protocol, "targets")
excluded_columns = need(protocol, "excluded_columns")
task = need(protocol, "task")
group_column = protocol.get("group_column", None)
if len(targets) != 1:
    raise ValueError("expected exactly one target")
target = targets[0]

base_excluded = set(targets) | set(excluded_columns)
if group_column in train.columns:
    base_excluded.add(group_column)
candidates = [c for c in train.columns if c not in base_excluded]
n = len(train)
key_like = []
for c in candidates:
    s = train[c]
    if n > 1 and s.nunique(dropna=True) == n and (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_integer_dtype(s)
    ):
        key_like.append(c)
features = [c for c in train.columns if c not in base_excluded and c not in key_like]

excluded_deriv = {}
if pd.api.types.is_numeric_dtype(train[target]):
    num_cands = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    for c in num_cands:
        if train[c].notna().all() and (train[c] == train[target]).all():
            excluded_deriv[c] = f"{c} equals {target}"
    eligible = [
        c for c in num_cands
        if c not in excluded_deriv and train[c].notna().all() and (train[c] >= 0).all()
    ]
    for r in (2, 3):
        for combo in combinations(eligible, r):
            if any(c in excluded_deriv for c in combo):
                continue
            s = train[list(combo)].sum(axis=1)
            if (s == train[target]).all():
                for c in combo:
                    excluded_deriv[c] = f"{combo} sum equals {target}"
    if excluded_deriv:
        for c in excluded_deriv:
            if c in features:
                features.remove(c)
        print("excluded derivations:", "; ".join(f"{c}: {r}" for c, r in excluded_deriv.items()))
    num_rem = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    if num_rem:
        corr = train[num_rem].corrwith(train[target]).abs()
        warn = [c for c in corr.index if corr[c] >= 0.98]
        if warn:
            print("corr>=0.98 kept:", warn)
else:
    for c in features:
        if train[c].astype(str).equals(train[target].astype(str)):
            excluded_deriv[c] = f"{c} equals {target}"
    if excluded_deriv:
        for c in excluded_deriv:
            if c in features:
                features.remove(c)
        print("excluded derivations:", "; ".join(f"{c}: {r}" for c, r in excluded_deriv.items()))

text_cols = [c for c in features if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
tokens = {"?", "-", "NA", "N/A"}
ph_cols = []
for c in text_cols:
    s = train[c].astype("string").str.strip()
    if (s.isin(tokens) | s.eq("")).any():
        ph_cols.append(c)
if ph_cols:
    print(f"placeholders found in {len(ph_cols)} columns: {ph_cols}")
    for c in ph_cols:
        s = train[c].astype("string").str.strip()
        train[c] = s.mask(s.isin(tokens) | s.eq(""), np.nan)
else:
    print("placeholders: none")

dt_cols = []
for c in list(features):
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dt_cols.append(c)
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        parsed = pd.to_datetime(s, errors="coerce")
        if parsed.notna().mean() > 0.9 and s.nunique(dropna=True) > 5:
            dt_cols.append(c)
for c in dt_cols:
    dt = pd.to_datetime(train[c], errors="coerce")
    for part, vals in {
        "year": dt.dt.year,
        "month": dt.dt.month,
        "day": dt.dt.day,
        "dow": dt.dt.dayofweek,
        "hour": dt.dt.hour,
    }.items():
        newc = f"{c}_{part}"
        train[newc] = vals
        features.append(newc)
    features.remove(c)

numeric_cols, cat_cols, dropped_cols = [], [], []
for c in features:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s):
        numeric_cols.append(c)
    elif pd.api.types.is_datetime64_any_dtype(s):
        dropped_cols.append(c)
    else:
        if s.nunique(dropna=True) > max(1, int(0.05 * n)):
            dropped_cols.append(c)
        else:
            cat_cols.append(c)

features = numeric_cols + cat_cols
X_train = train[features].copy()
width = len(numeric_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
dense = width <= 300

transformers = []
if numeric_cols:
    transformers.append((
        "num",
        Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
        numeric_cols,
    ))
if cat_cols:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    transformers.append((
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", missing_values=np.nan)),
            ("ohe", ohe),
        ]),
        cat_cols,
    ))

pre = ColumnTransformer(transformers, remainder="drop")
if dense:
    pre.set_output(transform="pandas")
    for _, trans, _ in transformers:
        if hasattr(trans, "set_output"):
            trans.set_output(transform="pandas")

mlca_runtime.fit_record(pre, X_train, "train")
mlca_runtime.save_model(pre, "preprocessor.pkl")
print(f"features={len(features)} numeric={len(numeric_cols)} cat={len(cat_cols)} dropped={len(dropped_cols)}")
print("saved preprocessor.pkl")
