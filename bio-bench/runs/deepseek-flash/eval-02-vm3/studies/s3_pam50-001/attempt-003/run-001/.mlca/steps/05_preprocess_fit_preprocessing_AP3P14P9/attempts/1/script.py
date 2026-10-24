import mlca_runtime, json
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
targets = list(proto["targets"]); excluded = list(proto["excluded_columns"])
group = proto.get("group_column")
y_col = targets[0]

train = pd.read_parquet(inp["train"]); val = pd.read_parquet(inp["val"])
n = len(train)

def is_text(s):
    return s.dtype == object or pd.api.types.is_string_dtype(s)

# RL-0023 placeholders
ph = {'?', '-', 'NA', 'N/A', 'n/a', 'na', ''}
ph_cols = []
for c in train.columns:
    if is_text(train[c]):
        s = train[c].astype(str).str.strip()
        m = s.isin(ph)
        if m.any():
            train[c] = s.where(~m, np.nan)
            if c in val.columns:
                sv = val[c].astype(str).str.strip()
                val[c] = sv.where(~sv.isin(ph), np.nan)
            ph_cols.append(c)
if ph_cols:
    print(f"placeholders in {len(ph_cols)} text cols")

# RL-0007 feature columns
excl = set(targets) | set(excluded) | ({group} if group else set())
feat_cols = [c for c in train.columns if c not in excl]
drop_keys = []
for c in feat_cols:
    s = train[c]
    if (pd.api.types.is_integer_dtype(s) or is_text(s)) and s.nunique(dropna=True) == n and n > 1:
        drop_keys.append(c)
feat_cols = [c for c in feat_cols if c not in drop_keys]

# RL-0008 derivation screen (categorical target -> exact equality only for text)
target_numeric = pd.api.types.is_numeric_dtype(train[y_col]) and not pd.api.types.is_bool_dtype(train[y_col])
deriv = []
if target_numeric:
    t = train[y_col].astype(float).to_numpy()
    numeric = [c for c in feat_cols if pd.api.types.is_numeric_dtype(train[c])]
    for c in numeric:
        if np.allclose(train[c].astype(float).to_numpy(), t, equal_nan=True):
            deriv.append((c, "equals_target"))
    nonneg = [c for c in numeric if (train[c].dropna() >= 0).all()]
    if 2 <= len(nonneg) <= 25:
        import itertools
        for k in (2, 3):
            for combo in itertools.combinations(nonneg, k):
                if np.allclose(train[list(combo)].sum(axis=1).to_numpy(), t, equal_nan=True):
                    for c in combo: deriv.append((c, "sum_to_target"))
                    break
    for c, r in deriv: print(f"excluded {c}: {r}")
    drop_deriv = {c for c, _ in deriv}
    feat_cols = [c for c in feat_cols if c not in drop_deriv]
    # correlation warnings
    warn = []
    for c in feat_cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            v = train[c].astype(float)
            if v.std() > 0:
                corr = np.corrcoef(v.fillna(v.median()), t)[0, 1]
                if abs(corr) >= 0.98: warn.append(c)
    if warn: print(f"corr>=0.98 with target: {warn}")
else:
    print("derivation screen: categorical target, skipped numeric checks")

# RL-0010 sort types
num_cols, cat_cols = [], []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    else:
        cat_cols.append(c)

transformers = []
if num_cols:
    transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                          ("sc", StandardScaler())]), num_cols))
if cat_cols:
    transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols))

ct = ColumnTransformer(transformers, remainder="drop")
ct.set_output(transform="pandas")

X_train = train[feat_cols]; y_train = train[y_col]
mlca_runtime.fit_record(ct, X_train, "train", y=y_train)
Xt = ct.transform(val[feat_cols])
print(f"fit ok: {len(feat_cols)} feats -> {Xt.shape[1]} encoded cols")

mlca_runtime.save_model(ct, "preprocessor.pkl")
