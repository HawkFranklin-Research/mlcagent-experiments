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
for k in ("targets", "excluded_columns", "task"):
    if k not in protocol:
        raise KeyError(f"protocol missing '{k}'; available: {list(protocol.keys())}")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")
forbidden = set(targets) | set(excluded)
if group_col:
    forbidden.add(group_col)
target = targets[0]
y_train = train[target]

feature_cols = []
for c in train.columns:
    if c in forbidden:
        continue
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        nn = s.dropna()
        if len(nn) and nn.nunique() == len(nn):
            continue
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        nn = s.dropna()
        if len(nn) and nn.nunique() == len(nn) and not set(nn.unique().tolist()).issubset({0, 1}):
            continue
    feature_cols.append(c)

ph_cols = []
for c in feature_cols:
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        st = s.dropna().astype(str).str.strip()
        if st.eq("").any() or st.isin(["?", "-", "NA", "N/A"]).any():
            ph_cols.append(c)
print(f"Placeholders found in {len(ph_cols)} text column(s)")

X_train = train[feature_cols].copy()
drop_reason = {}
num_feats = [c for c in feature_cols
             if pd.api.types.is_numeric_dtype(X_train[c]) and not pd.api.types.is_bool_dtype(X_train[c])]
if pd.api.types.is_numeric_dtype(y_train) and y_train.nunique() == 2:
    tvals = y_train.astype(float).values
    for c in num_feats:
        v = X_train[c].astype(float).values
        if np.array_equal(np.nan_to_num(v), np.nan_to_num(tvals)):
            drop_reason[c] = "exact equality with target"
    nonneg = [c for c in num_feats if c not in drop_reason
              and len(X_train[c].dropna()) == len(X_train) and (X_train[c].dropna() >= 0).all()]
    if 0 < len(nonneg) <= 16:
        found = False
        for r in (1, 2, 3):
            if found:
                break
            for combo in itertools.combinations(nonneg, r):
                s = sum(X_train[c].values.astype(float) for c in combo)
                if np.allclose(s, tvals):
                    for c in combo:
                        drop_reason[c] = "sum equals target"
                    found = True
                    break
if drop_reason:
    print("Excluded: " + "; ".join(f"{k} ({v})" for k, v in list(drop_reason.items())[:5]))
if drop_reason:
    feature_cols = [c for c in feature_cols if c not in drop_reason]
    X_train = X_train.drop(columns=list(drop_reason))
X_val = val[feature_cols].copy()

numeric_cols, cat_cols, high_card, date_cols = [], [], [], []
n = len(X_train)
for c in feature_cols:
    s = X_train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c); continue
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c); continue
    nn = s.dropna()
    if len(nn) and nn.nunique() > max(10, 0.05 * n):
        high_card.append(c)
    else:
        cat_cols.append(c)
if high_card:
    print(f"Dropped {len(high_card)} high-cardinality column(s)")
    X_train = X_train.drop(columns=high_card)
    X_val = X_val.drop(columns=high_card)
    feature_cols = [c for c in feature_cols if c not in high_card]
if date_cols:
    for c in date_cols:
        dt = pd.to_datetime(X_train[c], errors="coerce")
        dv = pd.to_datetime(X_val[c], errors="coerce")
        for part, fn in (("hour", lambda d: d.dt.hour),
                         ("dow", lambda d: d.dt.dayofweek),
                         ("month", lambda d: d.dt.month)):
            X_train[f"{c}_{part}"] = fn(dt).astype(float)
            X_val[f"{c}_{part}"] = fn(dv).astype(float)
            numeric_cols.append(f"{c}_{part}")
    X_train = X_train.drop(columns=date_cols)
    X_val = X_val.drop(columns=date_cols)

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
tfs = []
if numeric_cols:
    tfs.append(("num", num_pipe, numeric_cols))
if cat_cols:
    tfs.append(("cat", cat_pipe, cat_cols))
pre = ColumnTransformer(tfs, remainder="drop")
est_w = len(numeric_cols) + (sum(train[c].dropna().nunique() for c in cat_cols) if cat_cols else 0)
if est_w <= 300:
    pre.set_output(transform="pandas")
mlca_runtime.fit_record(pre, X_train, "train", y=y_train)
Xv = pre.transform(X_val)
print(f"Fit complete. train={X_train.shape} val_encoded={Xv.shape} est_width={est_w}")
mlca_runtime.save_model(pre, "preprocessor.pkl")
