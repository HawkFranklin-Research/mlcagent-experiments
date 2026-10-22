import mlca_runtime
import json
import itertools
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing {key}; available top-level keys: {list(protocol.keys())}")
targets = list(protocol["targets"])
excluded = set(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
target = targets[0]
y = train[target]

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
drop = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=False) == len(train):
            drop.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(train):
        if any(tok in c.lower() for tok in ("id", "key", "code")):
            drop.append(c)
for c in drop:
    feature_cols.remove(c)

derived = []
for c in list(feature_cols):
    if (train[c] == y).all():
        feature_cols.remove(c)
        derived.append(c)
if pd.api.types.is_numeric_dtype(y):
    ymax, ymin = y.max(), y.min()
    nonneg = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and train[c].dropna().min() >= 0]
    for k in (1, 2, 3):
        for combo in itertools.combinations(nonneg, k):
            if any(c not in feature_cols for c in combo):
                continue
            if train[list(combo)].max().sum() < ymax:
                continue
            if train[list(combo)].min().sum() > ymin:
                continue
            if (train[list(combo)].sum(axis=1) == y).all():
                for c in combo:
                    if c in feature_cols:
                        feature_cols.remove(c)
                derived.append(combo)
                nonneg = [c for c in nonneg if c in feature_cols]
if derived:
    print("Excluded derived features: " + ", ".join(str(x) for x in derived))

num_cols, cat_cols = [], []
for c in list(feature_cols):
    if pd.api.types.is_datetime64_any_dtype(train[c]):
        continue
    if pd.api.types.is_numeric_dtype(train[c]):
        num_cols.append(c)
    elif train[c].nunique(dropna=False) <= 0.05 * len(train):
        cat_cols.append(c)
feature_cols = num_cols + cat_cols
width = len(num_cols) + sum(train[c].nunique(dropna=False) for c in cat_cols)
wide = width > 300

transformers = []
if num_cols:
    transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols))
if cat_cols:
    transformers.append(("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=wide)),
    ]), cat_cols))

pre = ColumnTransformer(transformers, remainder="drop")
if not wide:
    pre.set_output(transform="pandas")
pre = mlca_runtime.fit_record(pre, train[feature_cols], "train", y=y) or pre
X_val = pre.transform(val[feature_cols])
print(f"Features {len(feature_cols)} numeric={len(num_cols)} categorical={len(cat_cols)}; val shape={X_val.shape}")
mlca_runtime.save_model(pre, "preprocessor.pkl")
