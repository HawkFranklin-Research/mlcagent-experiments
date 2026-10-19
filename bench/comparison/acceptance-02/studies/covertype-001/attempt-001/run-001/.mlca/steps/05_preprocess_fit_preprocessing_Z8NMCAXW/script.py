import json, itertools
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
protocol = json.loads(paths["_protocol"].read_text())

for k in ["targets", "excluded_columns", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing protocol key {k}; available top-level keys: {list(protocol)}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
drop_cols = set(targets) | set(excluded)
if group:
    drop_cols.add(group)
candidate = [c for c in train.columns if c not in drop_cols]

n_train = len(train)
key_like = []
for c in candidate:
    s = train[c]
    if s.dtype == object or str(s.dtype) in ("string", "string[python]"):
        if s.nunique(dropna=True) == n_train:
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        name = c.lower()
        if s.nunique(dropna=True) == n_train and any(t in name for t in ("id", "key", "code")):
            key_like.append(c)
features = [c for c in candidate if c not in key_like]
if key_like:
    print("Key-like excluded: " + ", ".join(key_like))

derived = {}
for tcol in targets:
    if tcol not in train.columns:
        continue
    t = train[tcol]
    for c in list(features):
        s = train[c]
        try:
            eq = s.notna().all() and t.notna().all() and (s == t).all()
        except Exception:
            eq = False
        if eq:
            derived[c] = f"equals target {tcol}"

tcol = targets[0]
if tcol in train.columns:
    t = train[tcol]
    if pd.api.types.is_numeric_dtype(t) and t.notna().all():
        nonneg = [c for c in features if pd.api.types.is_numeric_dtype(train[c]) and train[c].dropna().ge(0).all()]
        if nonneg:
            maxes = sorted([(train[c].max(), c) for c in nonneg], reverse=True)
            cands = [c for m, c in maxes if m <= t.max()]
            if cands:
                top = sum(sorted([train[c].max() for c in cands], reverse=True)[:3])
                if t.max() <= top:
                    for r in range(1, min(3, len(cands)) + 1):
                        for combo in itertools.combinations(cands, r):
                            s = train[list(combo)].sum(axis=1, skipna=False)
                            if s.notna().all() and (s == t).all():
                                for c in combo:
                                    derived.setdefault(c, "row sum equals target")
if derived:
    print("Excluded derivations: " + "; ".join(f"{c} ({r})" for c, r in derived.items()))
features = [c for c in features if c not in derived]

X_train = train[features].copy()
X_val = val[features].copy()

text_cols = [c for c in features if X_train[c].dtype == object or str(X_train[c].dtype) in ("string", "string[python]")]
ph = {}
for c in text_cols:
    s = X_train[c].astype("string").str.strip()
    mask = s.isin(["?", "-", "NA", "N/A", ""])
    if mask.any():
        ph[c] = int(mask.sum())
if ph:
    print("Placeholders found in " + str(len(ph)) + " columns: " + ", ".join(ph))
    for c in ph:
        for frame in (X_train, X_val):
            s = frame[c].astype("string").str.strip()
            frame[c] = s.where(~s.isin(["?", "-", "NA", "N/A", ""]), np.nan)

date_cols = [c for c in features if pd.api.types.is_datetime64_any_dtype(X_train[c])]
if date_cols:
    for c in date_cols:
        ref = X_train[c].min()
        parts = [("_month", X_train[c].dt.month, X_val[c].dt.month),
                 ("_dow", X_train[c].dt.dayofweek, X_val[c].dt.dayofweek),
                 ("_days", (X_train[c] - ref).dt.days, (X_val[c] - ref).dt.days)]
        for suffix, a, b in parts:
            name = c + suffix
            X_train[name] = a
            X_val[name] = b
            features.append(name)
    features = [c for c in features if c not in date_cols]
    X_train = X_train[features].copy()
    X_val = X_val[features].copy()

num_cols, cat_cols, high_card = [], [], []
for c in features:
    s = X_train[c]
    if pd.api.types.is_bool_dtype(s):
        X_train[c] = s.astype(int)
        X_val[c] = X_val[c].astype(int)
        num_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        nunq = s.nunique(dropna=True)
        if len(X_train) >= 20 and nunq > 0.05 * len(X_train):
            high_card.append(c)
        else:
            cat_cols.append(c)
if high_card:
    print("High-cardinality dropped: " + ", ".join(high_card))
features = [c for c in features if c not in high_card]
X_train = X_train[features].copy()
X_val = X_val[features].copy()

cat_width = sum(X_train[c].nunique(dropna=True) for c in cat_cols)
est_width = len(num_cols) + cat_width
onehot_sparse = est_width > 300
transformers = []
if num_cols:
    transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols))
if cat_cols:
    transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=onehot_sparse))]), cat_cols))
if not transformers:
    raise ValueError("No feature columns remain after screening")
preprocessor = ColumnTransformer(transformers, remainder="drop")
if est_width <= 300:
    preprocessor.set_output(transform="pandas")
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, "train")
val_out = preprocessor.transform(X_val)
print(f"Features {len(features)} (num {len(num_cols)}, cat {len(cat_cols)}); val transformed {val_out.shape[1]} cols")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
