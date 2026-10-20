import mlca_runtime, json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in protocol:
        raise ValueError(f"Missing '{k}'. Available: {sorted(protocol.keys())}")

train = pd.read_parquet(inp["train"]).copy()
val = pd.read_parquet(inp["val"]).copy()

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
y_col = targets[0]

drop = set(targets) | set(excluded)
if group and group in train.columns:
    drop.add(group)
cand = [c for c in train.columns if c not in drop]

# RL-0007: key-like drop
id_cols = []
for c in cand:
    s = train[c]; n = len(s)
    if s.nunique() != n or n < 2:
        continue
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        id_cols.append(c)
    elif pd.api.types.is_integer_dtype(s):
        try:
            if int(s.max() - s.min() + 1) == s.nunique():
                id_cols.append(c)
        except Exception:
            pass
if id_cols:
    print(f"Key-like dropped: {id_cols}")
features = [c for c in cand if c not in id_cols]

# RL-0023: placeholder detection + clean
phs = {"?", "-", "NA", "N/A"}
ph_cols = {}
for c in features:
    s = train[c]
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        sv = s.dropna().astype(str).str.strip()
        hit = [p for p in phs if (sv == p).any()]
        if hit:
            ph_cols[c] = hit
if ph_cols:
    toks = sorted({p for v in ph_cols.values() for p in v})
    print(f"Placeholders {toks} in {len(ph_cols)} text cols")
for c in features:
    s = train[c]
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        for df in (train, val):
            if c in df.columns:
                orig = df[c].notna()
                ss = df[c].astype(str).str.strip()
                ss = ss.where(~ss.isin(phs | {""}))
                df[c] = ss.where(orig)

# RL-0008: screen for target derivations
y_num = pd.api.types.is_numeric_dtype(train[y_col])
excl = []
for c in list(features):
    try:
        if (train[c] == train[y_col]).all():
            excl.append(c); features.remove(c)
    except Exception:
        pass
if y_num:
    yv = train[y_col].astype(float).values
    nc = [c for c in features if pd.api.types.is_numeric_dtype(train[c])
          and not pd.api.types.is_bool_dtype(train[c])
          and (train[c].fillna(0) >= 0).all()]
    done = False
    for r in (2, 3):
        for combo in combinations(nc, r):
            tot = sum(train[c].astype(float).values for c in combo)
            if np.allclose(tot, yv, equal_nan=True):
                excl.extend(combo)
                for c in combo:
                    if c in features:
                        features.remove(c)
                done = True; break
        if done:
            break
if excl:
    print(f"Excluded target-derived: {excl}")

# Correlation warning (numeric outcome only)
if y_num:
    yv2 = train[y_col].astype(float)
    w = []
    for c in features:
        if pd.api.types.is_numeric_dtype(train[c]):
            x = train[c].astype(float); m = x.notna() & yv2.notna()
            if m.sum() > 1 and x[m].std() > 0 and yv2[m].std() > 0:
                if abs(np.corrcoef(x[m], yv2[m])[0, 1]) >= 0.98:
                    w.append(c)
    if w:
        print(f"Corr>=0.98 kept: {w}")

# RL-0010: datetime derivation
dt_cols = [c for c in features if pd.api.types.is_datetime64_any_dtype(train[c])]
for c in dt_cols:
    for df in (train, val):
        if c in df.columns:
            d = pd.to_datetime(df[c], errors="coerce")
            df[f"{c}_year"] = d.dt.year
            df[f"{c}_month"] = d.dt.month
            df[f"{c}_day"] = d.dt.day
    features.remove(c)
    features.extend([f"{c}_year", f"{c}_month", f"{c}_day"])

num_cols, low_cat, high_cat = [], [], []
n = len(train); thr = max(10, int(0.05 * n))
for c in features:
    s = train[c]
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        (low_cat if s.nunique(dropna=True) <= thr else high_cat).append(c)

width = len(num_cols) + len(high_cat) + sum(train[c].nunique() for c in low_cat)
use_pd = width <= 300

trans = []
if num_cols:
    trans.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("sc", StandardScaler())]), num_cols))
if low_cat:
    trans.append(("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=not use_pd)),
    ]), low_cat))
if high_cat:
    trans.append(("hc", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), high_cat))

pre = ColumnTransformer(trans, remainder="drop")
if use_pd:
    pre.set_output(transform="pandas")

X_tr = train[features]
fitted = mlca_runtime.fit_record(pre, X_tr, partition="train", y=train[y_col])
v_out = fitted.transform(val[features])

mlca_runtime.save_model(fitted, "preprocessor.pkl")
mlca_runtime.record_metric("encoded_width", int(v_out.shape[1]), split="train")
mlca_runtime.record_metric("n_features_used", len(features), split="train")
print(f"Fitted {len(X_tr)} rows -> {v_out.shape[1]} encoded cols; val {v_out.shape}")
