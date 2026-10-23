import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

targets = protocol["targets"]
excluded = list(protocol.get("excluded_columns", []))
group_col = protocol.get("group_column", None)
task = protocol["task"]
primary_metric = task["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

drop_cols = set(targets) | set(excluded)
if group_col:
    drop_cols.add(group_col)
feat_candidates = [c for c in train.columns if c not in drop_cols]

# Key-like exclusion (RL-0007)
key_like = []
for c in feat_candidates:
    s = train[c]
    n = s.notna().sum()
    if n == 0:
        continue
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        if s.nunique(dropna=True) == n:
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        vals = s.dropna().unique()
        if len(vals) == n and not set(np.unique(vals)).issubset({0, 1}):
            key_like.append(c)

features = [c for c in feat_candidates if c not in key_like]

# Exact derivation screening (RL-0008)
y = train[targets[0]]
y_num = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
y_arr = y.astype(float).values if y_num else None

derived = []
for c in features[:]:
    if y_num:
        v = train[c].astype(float).values
        mask = ~(pd.isna(v) | pd.isna(y_arr))
        eq = (v[mask] == y_arr[mask]).all() and (pd.isna(v) == pd.isna(y_arr)).all()
    else:
        eq = (train[c].astype(str).values == y.astype(str).values).all()
    if eq:
        derived.append(c)

if y_num and (y_arr >= 0).all():
    small_int = [c for c in features if (
        pd.api.types.is_numeric_dtype(train[c]) and
        (train[c].dropna() >= 0).all() and
        (train[c].dropna() <= 10).all() and
        (train[c].dropna().round() == train[c].dropna()).all()
    )]
    cand = {c: train[c].fillna(0).values.astype(int) for c in small_int}
    keys = list(cand.keys())
    for i, ci in enumerate(keys):
        for cj in keys[i+1:]:
            s = cand[ci] + cand[cj]
            if (s == y_arr).all() and ci not in derived and cj not in derived:
                derived.extend([ci, cj])

features = [c for c in features if c not in derived]

# Placeholder check (RL-0023)
ph_tokens = {"?", "-", "NA", "N/A", ""}
ph_cols = []
for c in features:
    s = train[c]
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        stripped = s.astype(str).str.strip()
        for t in ph_tokens:
            cnt = (stripped == t).sum() if t != "" else (stripped == "").sum()
            if cnt > 0:
                ph_cols.append(c)
                break
if ph_cols:
    for c in ph_cols:
        for df in (train, val):
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace(list(ph_tokens - {""}), np.nan).replace("", np.nan)

# Sort feature types (RL-0010)
numeric_cols, cat_cols, high_card, datetime_cols = [], [], [], []
n_rows = len(train)
threshold = 0.05 * n_rows

for c in features:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        datetime_cols.append(c)
    elif pd.api.types.is_bool_dtype(s):
        train[c] = train[c].astype(int)
        val[c] = val[c].astype(int)
        numeric_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    else:
        nun = s.nunique(dropna=True)
        if nun <= max(2, threshold):
            cat_cols.append(c)
        else:
            high_card.append(c)

for c in datetime_cols:
    dt_train = pd.to_datetime(train[c], errors="coerce")
    dt_val = pd.to_datetime(val[c], errors="coerce")
    for part, func in [("year", lambda x: x.dt.year), ("month", lambda x: x.dt.month),
                       ("day", lambda x: x.dt.day), ("hour", lambda x: x.dt.hour),
                       ("dow", lambda x: x.dt.dayofweek)]:
        nc = f"{c}_{part}"
        train[nc] = func(dt_train).astype(float)
        val[nc] = func(dt_val).astype(float)
        numeric_cols.append(nc)

features = numeric_cols + cat_cols

# Build preprocessor with correct one-hot sparsity (RL-0009)
est_width = len(numeric_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
ohe_sparse = est_width > 300

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=ohe_sparse)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, cat_cols),
    ],
    remainder="drop",
)
if est_width <= 300:
    preprocessor.set_output(transform="pandas")

mlca_runtime.fit_record(preprocessor, train[features], "train")
_ = preprocessor.transform(val[features])
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

print(f"Features: {len(features)} ({len(numeric_cols)} numeric, {len(cat_cols)} categorical); "
      f"dropped {len(key_like)} key-like, {len(derived)} derived, {len(high_card)} high-card")
if ph_cols:
    print(f"Placeholders found in {len(ph_cols)} columns")
