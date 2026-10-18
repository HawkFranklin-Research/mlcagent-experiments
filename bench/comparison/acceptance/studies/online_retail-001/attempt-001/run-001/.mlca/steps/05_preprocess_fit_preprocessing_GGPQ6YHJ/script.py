import mlca_runtime, json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
train = pd.read_parquet(paths["train"])

targets = list(protocol.get("targets", []) or [])
excluded = list(protocol.get("excluded_columns", []) or [])
group_col = protocol.get("group_column", None)
drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)

cands = [c for c in train.columns if c not in drop]
feat = []
for c in cands:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if s.dtype == object or str(s.dtype).startswith("str"):
        nn = s.dropna().astype(str).str.strip()
        if len(nn) > 5 and nn.nunique() == len(nn):
            print(f"drop key-like column: {c}")
            continue
    feat.append(c)

PH = {"?", "-", "NA", "N/A", ""}
X = train[feat].copy()
ph_cols = []
for c in feat:
    if pd.api.types.is_numeric_dtype(X[c]):
        continue
    sm = X[c].astype(object).where(X[c].notna(), np.nan)
    sm = sm.map(lambda v: v.strip() if isinstance(v, str) else v)
    if sm.map(lambda v: isinstance(v, str) and v in PH).any():
        ph_cols.append(c)
    X[c] = sm.map(lambda v: np.nan if isinstance(v, str) and v in PH else v)
if ph_cols:
    print(f"placeholder tokens ({len(ph_cols)} cols): {','.join(ph_cols[:5])}")

num_cols, cat_cols = [], []
n = len(X)
for c in feat:
    if c not in X.columns:
        continue
    s = X[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    else:
        nu = s.nunique(dropna=True)
        if nu > max(20, int(0.05 * n)):
            print(f"drop high-card col: {c}")
            continue
        cat_cols.append(c)

log_cols = [c for c in num_cols
            if X[c].dropna().min() >= 0 and abs(X[c].skew()) > 2]
plain_num = [c for c in num_cols if c not in log_cols]

est_width = len(num_cols) + sum(X[c].nunique(dropna=True) for c in cat_cols)
sparse = est_width > 300

num_log = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("sc", StandardScaler()),
])
num_pl = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat_pl = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)),
])

pre = ColumnTransformer(
    [("nl", num_log, log_cols),
     ("nn", num_pl, plain_num),
     ("cat", cat_pl, cat_cols)],
    remainder="drop",
    sparse_threshold=1.0 if sparse else 0.0,
)
if not sparse:
    pre.set_output(transform="pandas")

fitted = mlca_runtime.fit_record(pre, X, "train")
mlca_runtime.save_model(fitted, "preprocessor.pkl")
print(f"features={len(feat)} num={len(num_cols)} log={len(log_cols)} cat={len(cat_cols)} sparse={sparse}")
