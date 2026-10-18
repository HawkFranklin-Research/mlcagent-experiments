import json
import pandas as pd
import numpy as np
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
target = protocol["targets"][0]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")
n = len(train)

def keylike(s):
    if pd.api.types.is_integer_dtype(s) or s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=False) == n
    return False

cand = [c for c in train.columns if c not in protocol["targets"] and c not in excluded and c != group and not keylike(train[c])]
y = train[target].to_numpy()
leaks = set()
for c in cand:
    if np.array_equal(train[c].fillna(-999).to_numpy(), y):
        leaks.add(c); print(f"excluded {c}: equals target")
nonneg = [c for c in cand if pd.api.types.is_numeric_dtype(train[c]) and (train[c] >= 0).all()]
for i in range(len(nonneg)):
    for j in range(i+1, len(nonneg)):
        if np.array_equal((train[nonneg[i]].fillna(0) + train[nonneg[j]].fillna(0)).to_numpy(), y):
            leaks.update([nonneg[i], nonneg[j]]); print(f"excluded {nonneg[i]},{nonneg[j]}: sum target")
        for k in range(j+1, len(nonneg)):
            if np.array_equal((train[nonneg[i]].fillna(0) + train[nonneg[j]].fillna(0) + train[nonneg[k]].fillna(0)).to_numpy(), y):
                leaks.update([nonneg[i], nonneg[j], nonneg[k]]); print(f"excluded {nonneg[i]},{nonneg[j]},{nonneg[k]}: sum target")
features = [c for c in cand if c not in leaks]

if pd.api.types.is_numeric_dtype(train[target]) and train[target].nunique() > 1:
    warn = [c for c in features if pd.api.types.is_numeric_dtype(train[c]) and abs(np.corrcoef(train[c].fillna(train[c].median()).astype(float), y.astype(float))[0,1]) >= 0.98]
    if warn: print(f"high corr with target (kept): {warn}")

ph = ['?', '-', 'NA', 'N/A']
ph_cols = []
for c in features:
    if pd.api.types.is_string_dtype(train[c]) or train[c].dtype == object:
        s = train[c].astype(str).str.strip()
        if s.isin(ph).any() or (s == '').any():
            ph_cols.append(c)
if ph_cols: print(f"placeholders in {len(ph_cols)} cols: {ph_cols}")

max_cat = max(1, int(0.05 * n))
num_cols, cat_cols = [], []
for c in features:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        if s.nunique(dropna=True) <= max_cat:
            cat_cols.append(c)
        else:
            print(f"dropped {c}: {s.nunique(dropna=True)} unique > {max_cat}")

num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
num_pipe.set_output(transform="pandas")
cat_pipe.set_output(transform="pandas")
pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
pre.set_output(transform="pandas")
mlca_runtime.fit_record(pre, train[features], "train", y=train[target])
transformed = pre.transform(val[features])
print(f"features {len(features)}: {len(num_cols)} num, {len(cat_cols)} cat; val shape {transformed.shape}")
mlca_runtime.save_model(pre, "preprocessor.pkl")
