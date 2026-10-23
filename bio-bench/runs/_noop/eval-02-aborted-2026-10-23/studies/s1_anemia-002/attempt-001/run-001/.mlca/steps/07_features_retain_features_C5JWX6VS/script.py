import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
for key in ("targets", "excluded_columns"):
    if key not in protocol:
        raise KeyError(f"Missing {key}. Available top-level keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
train = pd.read_parquet(paths["train"])

with open(paths["06_preprocess_fit_preprocessing_FTEQSYVR/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

expected = getattr(pre, "feature_names_in_", None)
if expected is not None:
    feature_cols = list(expected)
    missing = [c for c in feature_cols if c not in train.columns]
    if missing:
        raise KeyError(f"Missing expected feature columns: {missing}")
    bad = [c for c in feature_cols if c in targets or c in excluded or (group and c == group)]
    if bad:
        raise ValueError(f"Target/excluded/group columns in preprocessor input: {bad}")
else:
    feature_cols = []
    for c in train.columns:
        if c in targets or c in excluded or (group and c == group):
            continue
        s = train[c]
        lc = str(c).lower()
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s) and not s.isna().any():
            continue
        if pd.api.types.is_integer_dtype(s) and ("id" in lc or lc in ("index", "row_id", "record_id")) and s.nunique(dropna=False) == len(s):
            continue
        feature_cols.append(c)

if not hasattr(pre, "transform"):
    num_cols, cat_cols = [], []
    for c in feature_cols:
        s = train[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            raise ValueError("Datetime columns require the saved preprocessor with date handling.")
        (num_cols if pd.api.types.is_numeric_dtype(s) else cat_cols).append(c)

    n = len(train)
    keep_cat = [c for c in cat_cols if train[c].nunique(dropna=True) <= max(1, int(0.05 * n))]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    transforms = []
    if num_cols:
        transforms.append(("num", num_pipe, num_cols))
    if keep_cat:
        transforms.append(("cat", cat_pipe, keep_cat))
    pre = ColumnTransformer(transforms, remainder="drop")
    feature_cols = num_cols + keep_cat

pipeline = Pipeline([("preprocessor", pre), ("selector", "passthrough")])
X = train[feature_cols]
y = train[targets[0]] if len(targets) == 1 else train[targets]
fitted = mlca_runtime.fit_record(pipeline, X, "train", y)
mlca_runtime.save_model(fitted, "selector_all.pkl")
print(f"train rows={len(train)} features={len(feature_cols)}")
print("saved selector_all.pkl")
