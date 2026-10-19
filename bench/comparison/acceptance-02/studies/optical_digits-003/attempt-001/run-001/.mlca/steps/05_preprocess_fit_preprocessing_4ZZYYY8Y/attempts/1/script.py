import mlca_runtime, json, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
protocol = json.loads(inp["_protocol"].read_text())

for k in ("targets", "excluded_columns", "task"):
    if k not in protocol:
        raise KeyError(f"missing {k}; available top-level keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
all_cols = list(train.columns)
features = [c for c in all_cols if c not in targets and c not in excluded and c != group]

def key_like(s):
    if s.dtype == object and s.nunique() == len(s):
        return True
    if pd.api.types.is_integer_dtype(s) and s.nunique() == len(s) and s.min() == 0 and s.max() == len(s) - 1:
        return True
    return False

keys = [c for c in features if key_like(train[c])]
if keys:
    print("key-like dropped:", keys)
features = [c for c in features if c not in keys]

numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
dt_cols = [c for c in features if pd.api.types.is_datetime64_any_dtype(train[c])]
if dt_cols:
    for c in dt_cols:
        train[c + "_month"] = train[c].dt.month
        train[c + "_dow"] = train[c].dt.dayofweek
    features = [c for c in features if c not in dt_cols] + [c + "_month" for c in dt_cols] + [c + "_dow" for c in dt_cols]
    numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]

cat_cols = [c for c in features if c not in numeric_cols and c not in dt_cols]
if cat_cols:
    ph = {"?", "-", "NA", "N/A", ""}
    found = [c for c in cat_cols if train[c].astype(str).str.strip().isin(ph).any()]
    if found:
        print("placeholder columns:", found)
    high = [c for c in cat_cols if train[c].nunique() > 0.05 * len(train)]
    if high:
        print("high-cardinality categoricals dropped:", high)
        features = [c for c in features if c not in high]
        cat_cols = [c for c in cat_cols if c not in high]

log_cols = [c for c in numeric_cols if train[c].min() >= 0 and train[c].skew() > 2]
plain_cols = [c for c in numeric_cols if c not in log_cols]

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
log_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("log", FunctionTransformer(np.log1p)), ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
for p in (num_pipe, log_pipe, cat_pipe):
    p.set_output(transform="pandas")

trans = []
if log_cols:
    trans.append(("log", log_pipe, log_cols))
if plain_cols:
    trans.append(("num", num_pipe, plain_cols))
if cat_cols:
    trans.append(("cat", cat_pipe, cat_cols))

ct = ColumnTransformer(trans, remainder="drop")
ct.set_output(transform="pandas")
X = train.loc[:, features].copy()
ct = mlca_runtime.fit_record(ct, X, "train")
mlca_runtime.save_model(ct, "preprocessor.pkl")
print("features:", len(features), "log1p:", len(log_cols), "categorical:", len(cat_cols), "shape:", X.shape)
