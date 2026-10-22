import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out_dir = mlca_runtime.output_dir()

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

target = targets[0]
y_train = train[target]
y_val = val[target]

# key-like column detection
def is_key_like(s):
    if s.dtype == object:
        nun = s.nunique(dropna=True)
        return nun == len(s) and nun > 0 and s.astype(str).str.len().nunique() == 1 and s.astype(str).str.len().iloc[0] > 6
    return False

feat_cols = []
for c in train.columns:
    if c in targets or c in excluded:
        continue
    if group_col and c == group_col:
        continue
    if is_key_like(train[c]):
        continue
    feat_cols.append(c)

# derivation screen: exact equality or sums of up to 3 non-negative columns
excluded_feats = []
for c in list(feat_cols):
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        try:
            if s.fillna(-1e18).equals(y_train.astype(float)):
                excluded_feats.append((c, "equals target"))
                continue
        except Exception:
            pass

numeric_candidates = [c for c in feat_cols if pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique() > 2]
non_neg = [c for c in numeric_candidates if (train[c].dropna() >= 0).all()]
# limited 2-col sum check
for i in range(min(len(non_neg), 20)):
    for j in range(i + 1, min(len(non_neg), 20)):
        comb = train[non_neg[i]].fillna(0) + train[non_neg[j]].fillna(0)
        if comb.equals(y_train.astype(float)):
            for c in (non_neg[i], non_neg[j]):
                if c not in [x[0] for x in excluded_feats]:
                    excluded_feats.append((c, "sum equals target"))

for c, reason in excluded_feats:
    print(f"Excluded {c}: {reason}")
    if c in feat_cols:
        feat_cols.remove(c)

# sort types
num_cols, cat_cols = [], []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        n = s.nunique(dropna=True)
        if n <= 5 or n < 0.05 * len(train):
            cat_cols.append(c)
        else:
            cat_cols.append(c)

num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
], remainder="drop")

X_train = train[feat_cols].copy()
X_val = val[feat_cols].copy()

fitted = mlca_runtime.fit_record(pre, X_train, "train", y=y_train)
X_val_t = fitted.transform(X_val)
print(f"features={len(feat_cols)} num={len(num_cols)} cat={len(cat_cols)} val_encoded={X_val_t.shape}")

with open(out_dir / "preprocessor.pkl", "wb") as f:
    pickle.dump(fitted, f)

mlca_runtime.save_model(fitted, "preprocessor.pkl")
mlca_runtime.record_metric("n_features", len(feat_cols), "train")
