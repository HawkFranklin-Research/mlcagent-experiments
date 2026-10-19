import json
import mlca_runtime
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def require(d, keys):
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"Missing keys {missing}; available: {list(d.keys())}")

require(protocol, ["targets", "excluded_columns", "task"])
targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
missing_targets = [c for c in targets if c not in train.columns]
if missing_targets:
    raise KeyError(f"Targets not in train: {missing_targets}")

base_exclude = set(targets) | excluded
if group:
    base_exclude.add(group)
candidate_cols = [c for c in train.columns if c not in base_exclude]

feature_cols = []
for c in candidate_cols:
    s = train[c]
    lc = c.lower()
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if any(tok in lc for tok in ("id", "key", "index", "patient", "record", "subject")):
        continue
    nunique = s.nunique(dropna=True)
    nonnull = s.notna().sum()
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and nonnull > 1 and nunique == nonnull:
        continue
    if pd.api.types.is_integer_dtype(s) and nonnull > 1 and nunique == nonnull and any(tok in lc for tok in ("id", "key", "index", "code")):
        continue
    feature_cols.append(c)

num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]
threshold = max(2, int(0.05 * len(train)))
low_cat = [c for c in cat_cols if train[c].nunique(dropna=True) <= threshold]
feature_cols = [c for c in feature_cols if c in set(num_cols) | set(low_cat)]

num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
for pipe in (num_pipe, cat_pipe):
    pipe.set_output(transform="pandas")

transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if low_cat:
    transformers.append(("cat", cat_pipe, low_cat))

pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
pre.set_output(transform="pandas")

model = Pipeline([
    ("pre", pre),
    ("clf", MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))),
])
model.set_output(transform="pandas")

X = train[feature_cols]
y = train[targets]
fitted = mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(fitted, "alternative_model.pkl")
print(f"train rows={len(train)} features={len(feature_cols)} targets={len(targets)}")
