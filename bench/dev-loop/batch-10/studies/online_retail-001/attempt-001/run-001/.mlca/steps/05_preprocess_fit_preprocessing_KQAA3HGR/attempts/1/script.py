import mlca_runtime, json, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for k in ("targets", "excluded_columns", "group_column", "task"):
    if k not in protocol:
        raise KeyError(f"missing {k}; available: {list(protocol.keys())}")
targets = list(protocol["targets"]); excluded = list(protocol["excluded_columns"])
group_col = protocol["group_column"]
train = pd.read_parquet(inputs["train"])

all_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
key_like, high_card, date_cols = [], [], []
n = len(train)
for c in all_cols:
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        nn = s.dropna()
        if len(nn) > 0 and nn.nunique() == len(nn):
            key_like.append(c)
for c in all_cols:
    if c in key_like: continue
    s = train[c]
    if pd.api.types.is_integer_dtype(s):
        nn = s.dropna()
        if len(nn) > 0 and nn.nunique() == len(nn):
            key_like.append(c)
features = [c for c in all_cols if c not in key_like]
for c in features:
    s = train[c]
    if (s.dtype == object or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) > 0.05 * n:
        high_card.append(c)
features = [c for c in features if c not in high_card]
for c in features:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c); continue
    if s.dtype == object:
        samp = s.dropna().head(20)
        if len(samp) > 0:
            try:
                pd.to_datetime(samp, errors="raise"); date_cols.append(c)
            except Exception:
                pass
features = [c for c in features if c not in date_cols]

placeholders = ["?", "-", "NA", "N/A", ""]
ph_cols = {}
for c in features:
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        stripped = s.where(s.isna(), s.astype(str).str.strip())
        mask = stripped.isin(placeholders)
        cnt = int(mask.sum())
        if cnt > 0: ph_cols[c] = cnt
        train[c] = stripped.mask(mask, np.nan)

num_cols, cat_cols = [], []
for c in features:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    else:
        cat_cols.append(c)
log_cols = []
for c in num_cols:
    s = train[c].dropna()
    if len(s) > 0 and (s >= 0).all() and s.skew() > 2:
        log_cols.append(c)
plain_num = [c for c in num_cols if c not in log_cols]

num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
log_pipe = Pipeline([("log", FunctionTransformer(np.log1p, validate=False)),
                     ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                     ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
parts = []
if plain_num: parts.append(("num", num_pipe, plain_num))
if log_cols: parts.append(("log_num", log_pipe, log_cols))
if cat_cols: parts.append(("cat", cat_pipe, cat_cols))
ct = ColumnTransformer(parts, remainder="drop")
for p in (num_pipe, log_pipe, cat_pipe, ct):
    try: p.set_output(transform="pandas")
    except Exception: pass

X_train = train[features].copy()
mlca_runtime.fit_record(ct, X_train, partition="train")
Xt = ct.transform(X_train)
drops = key_like + high_card + date_cols
print(f"dropped {len(drops)} cols {drops}")
print(f"placeholders: {ph_cols if ph_cols else 'none'}")
print(f"features {len(features)} -> {Xt.shape[1]} output columns")
mlca_runtime.save_model(ct, "preprocessor.pkl")
