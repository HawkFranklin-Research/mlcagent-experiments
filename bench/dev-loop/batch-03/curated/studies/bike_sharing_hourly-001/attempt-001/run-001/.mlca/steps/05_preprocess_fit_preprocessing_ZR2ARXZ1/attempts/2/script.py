import json
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())
for k in ("targets", "excluded_columns", "group_column", "task"):
    if k not in protocol:
        raise KeyError(f"Missing {k}. Available: {list(protocol.keys())}")
targets, excluded, group_col, task = protocol["targets"], protocol["excluded_columns"], protocol["group_column"], protocol["task"]
drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)

feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if (s.dtype == object or str(s.dtype).startswith("str")) and s.nunique(dropna=True) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
        continue
    feature_cols.append(c)

for c in list(feature_cols):
    s = train[c]
    if s.dtype == object or str(s.dtype).startswith("str"):
        if pd.to_datetime(s, errors="coerce").notna().mean() > 0.9:
            feature_cols.remove(c)

y = train[targets[0]]
excluded_feats = []
for c in list(feature_cols):
    if train[c].equals(y):
        excluded_feats.append((c, "exact equality with target"))
        feature_cols.remove(c)

numeric_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and (train[c] >= 0).all()]
for i, c1 in enumerate(numeric_feats):
    if c1 not in feature_cols:
        continue
    for c2 in numeric_feats[i+1:]:
        if c2 not in feature_cols:
            continue
        if (train[c1] + train[c2]).equals(y):
            excluded_feats.append((c1, f"sum with {c2} equals target"))
            feature_cols.remove(c1)
            break

corr_warnings = []
if task.get("type") == "regression":
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train[c]) and abs(train[c].corr(y)) >= 0.98:
            corr_warnings.append(c)

placeholders = {"?", "-", "NA", "N/A", ""}
placeholder_cols = []
for c in feature_cols:
    s = train[c]
    if s.dtype == object or str(s.dtype).startswith("str"):
        if s.astype(str).str.strip().isin(placeholders).any():
            placeholder_cols.append(c)

threshold = 0.05 * len(train)
numeric_cols, cat_cols = [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    elif s.nunique(dropna=True) <= threshold:
        cat_cols.append(c)

n_out = len(numeric_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=(n_out > 300))
numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)])
preprocessor = ColumnTransformer([("num", numeric_pipe, numeric_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")
if n_out <= 300:
    preprocessor.set_output(transform="pandas")
X_train = train[feature_cols]
mlca_runtime.fit_record(preprocessor, X_train, "train")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

if excluded_feats:
    print(f"Excluded features: {excluded_feats}")
if corr_warnings:
    print(f"Correlation warnings (|r|>=0.98): {corr_warnings}")
if placeholder_cols:
    print(f"Placeholder tokens in {len(placeholder_cols)} columns: {placeholder_cols}")
print(f"Fitted preprocessor on {X_train.shape[0]} rows, {len(feature_cols)} features")
