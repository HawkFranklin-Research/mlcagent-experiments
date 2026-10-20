import mlca_runtime, json
import pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
if "targets" not in protocol:
    raise KeyError(f"Missing 'targets'. Available top-level keys: {list(protocol.keys())}")
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")
train = pd.read_parquet(inp["train"])

def is_key(c):
    s = train[c]
    if pd.api.types.is_integer_dtype(s):
        u = np.sort(s.dropna().unique()); n = s.notna().sum()
        if len(u) == n and (np.array_equal(u, np.arange(len(u))) or np.array_equal(u, np.arange(1, len(u) + 1))):
            return True
        return any(k in c.lower() for k in ("id", "key", "code"))
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=True) == s.notna().sum()
    return False

excl = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in excl and not is_key(c)]
n = len(train)
numeric_cols, cat_cols, dt_cols = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dt_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=True) <= max(20, 0.05 * n):
            cat_cols.append(c)
    else:
        numeric_cols.append(c)

if dt_cols:
    train = train.copy()
    for c in dt_cols:
        train[c + "__ts"] = train[c].astype("int64") / 1e9
        feature_cols.remove(c)
        numeric_cols.append(c + "__ts")

n_out = len(numeric_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
use_pandas = n_out <= 300
onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=not use_pandas)
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]) if numeric_cols else "passthrough"
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)]) if cat_cols else "passthrough"
transformers = []
if numeric_cols: transformers.append(("num", num_pipe, numeric_cols))
if cat_cols: transformers.append(("cat", cat_pipe, cat_cols))
pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
if use_pandas:
    pre.set_output(transform="pandas")
clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
pipeline = Pipeline([("pre", pre), ("clf", clf)])
if use_pandas:
    pipeline.set_output(transform="pandas")
model = mlca_runtime.fit_record(pipeline, train[feature_cols], "train", y=train[targets])
saved = mlca_runtime.save_model(model, "alternative_model.pkl")
print(f"features={len(feature_cols)} numeric={len(numeric_cols)} categorical={len(cat_cols)} width={n_out}")
print(f"saved {saved.name}")
