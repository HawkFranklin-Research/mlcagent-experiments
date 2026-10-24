import json
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
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", []) or []
group_col = protocol.get("group_column")
y_col = targets[0]
y_train = train[y_col]

drop_set = set(targets) | set(excluded)
if group_col:
    drop_set.add(group_col)
candidates = [c for c in train.columns if c not in drop_set]

n = len(train)
features, key_drops, deriv_drops = [], [], []
for c in candidates:
    s = train[c]
    if s.nunique(dropna=False) == n and (
        pd.api.types.is_integer_dtype(s)
        or pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
    ):
        key_drops.append(c)
        continue
    if s.astype(str).eq(y_train.astype(str)).all():
        deriv_drops.append(c)
        continue
    features.append(c)

num_cols, cat_cols, high_card = [], [], []
for c in features:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        high_card.append(c)
        continue
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
        continue
    nun = s.nunique(dropna=True)
    if nun > 5 and nun > 0.05 * n:
        high_card.append(c)
    else:
        cat_cols.append(c)

phs = {"", "?", "-", "NA", "N/A", "na", "n/a"}
ph_cols = []
for c in cat_cols:
    st = train[c].astype(str).str.strip()
    if st.isin(phs).any():
        ph_cols.append(c)
        train[c] = st.where(~st.isin(phs), other=None)
        vs = val[c].astype(str).str.strip()
        val[c] = vs.where(~vs.isin(phs), other=None)

final_features = num_cols + cat_cols
X_train = train[final_features]
X_val = val[final_features]

cat_width = sum(train[c].nunique(dropna=True) for c in cat_cols) if cat_cols else 0
use_dense = (len(num_cols) + cat_width) <= 300

transformers = []
if num_cols:
    transformers.append(("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]), num_cols))
if cat_cols:
    transformers.append(("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=not use_dense)),
    ]), cat_cols))

pre = ColumnTransformer(transformers, remainder="drop")
if use_dense:
    pre.set_output(transform="pandas")

fitted = mlca_runtime.fit_record(pre, X_train, "train")
Xt = fitted.transform(X_val)

mlca_runtime.save_model(fitted, "preprocessor.pkl")
mlca_runtime.record_metric("n_features", float(len(final_features)), "train")
print(f"features={len(final_features)} num={len(num_cols)} cat={len(cat_cols)}")
print(f"drop key={key_drops} derived={deriv_drops} highcard={high_card}")
print(f"placeholder cols={ph_cols}")
print(f"train={X_train.shape} val_enc={Xt.shape}")
