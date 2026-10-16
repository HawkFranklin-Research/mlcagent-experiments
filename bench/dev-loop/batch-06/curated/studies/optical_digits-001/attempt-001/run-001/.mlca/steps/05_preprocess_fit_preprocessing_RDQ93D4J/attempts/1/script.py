import mlca_runtime, json, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
ins = mlca_runtime.inputs()
proto = json.loads(ins["_protocol"].read_text())
targets = proto["targets"]
excluded = proto["excluded_columns"]
group = proto.get("group_column")

train = pd.read_parquet(ins["train"])
n = len(train)

drop = set(targets) | set(excluded)
if group:
    drop.add(group)

keylike = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if (pd.api.types.is_integer_dtype(s) or s.dtype == object) and n > 0 \
            and s.nunique(dropna=True) == n:
        keylike.append(c)

feat = [c for c in train.columns if c not in drop and c not in keylike]

num_cols, cat_cols, hi_cols = [], [], []
for c in feat:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        nu = s.nunique(dropna=True)
        if nu <= max(5, int(0.05 * n)):
            cat_cols.append(c)
        else:
            hi_cols.append(c)

log_cols, plain_num = [], []
for c in num_cols:
    v = train[c].dropna()
    if len(v) and (v >= 0).all() and v.skew() > 2:
        log_cols.append(c)
    else:
        plain_num.append(c)

parts = []
if log_cols:
    parts.append(("log", Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]), log_cols))
if plain_num:
    parts.append(("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]), plain_num))
if cat_cols:
    parts.append(("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), cat_cols))

ct = ColumnTransformer(parts, remainder="drop")

width = len(num_cols) + sum(int(train[c].nunique(dropna=True)) + 1 for c in cat_cols)
if width <= 300:
    ct.set_output(transform="pandas")

ct = mlca_runtime.fit_record(ct, train[feat], "train")

mlca_runtime.save_model(ct, "preprocessor.pkl")
mlca_runtime.record_metric("n_features_used", float(len(feat)), "train")
mlca_runtime.record_metric("n_keylike_dropped", float(len(keylike)), "train")
mlca_runtime.record_metric("n_high_card_dropped", float(len(hi_cols)), "train")

print(f"features={len(feat)} numeric={len(num_cols)} log1p={len(log_cols)} cat={len(cat_cols)}")
if keylike:
    print("keylike dropped: " + ",".join(keylike))
if hi_cols:
    print("high-cardinality dropped: " + ",".join(hi_cols))
print(f"output width ~{width}")
