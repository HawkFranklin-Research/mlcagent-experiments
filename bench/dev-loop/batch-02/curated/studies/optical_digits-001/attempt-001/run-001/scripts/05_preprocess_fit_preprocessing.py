import mlca_runtime, json, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise ValueError(f"Missing keys {missing}; available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
task = protocol["task"]

all_cols = list(train.columns)
exclude = set(targets) | set(excluded)
if group is not None:
    exclude.add(group)
for c in all_cols:
    if c in exclude:
        continue
    s = train[c]
    if s.nunique(dropna=True) == len(train) and not pd.api.types.is_float_dtype(s):
        exclude.add(c)
features = [c for c in all_cols if c not in exclude]

# RL-0008: no targets in this representation task, so no derivation exclusions.
# If targets were present, exact-derivation screening would run here.

numeric_cols, cat_cols = [], []
for c in features:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    else:
        nunique = s.nunique(dropna=True)
        if nunique <= 0.05 * len(train):
            cat_cols.append(c)

heavy_cols = []
for c in numeric_cols:
    s = train[c].dropna()
    if len(s) > 0 and (s >= 0).all() and s.skew() > 2:
        heavy_cols.append(c)
regular_num = [c for c in numeric_cols if c not in heavy_cols]

expected_width = len(regular_num) + len(heavy_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
sparse = expected_width > 300
transformers = []

if regular_num:
    p = Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    if not sparse:
        p.set_output(transform="pandas")
    transformers.append(("num", p, regular_num))

if heavy_cols:
    p = Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("log", FunctionTransformer(np.log1p, validate=False)),
                  ("scale", StandardScaler())])
    if not sparse:
        p.set_output(transform="pandas")
    transformers.append(("num_log", p, heavy_cols))

if cat_cols:
    p = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                  ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse))])
    if not sparse:
        p.set_output(transform="pandas")
    transformers.append(("cat", p, cat_cols))

preprocessor = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
if not sparse:
    preprocessor.set_output(transform="pandas")

X_train = train[features]
mlca_runtime.fit_record(preprocessor, X_train, partition="train", y=None)
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

print(f"features={len(features)} numeric={len(numeric_cols)} cat={len(cat_cols)} heavy={len(heavy_cols)}")
print(f"encoded_width={expected_width} sparse={sparse}")
