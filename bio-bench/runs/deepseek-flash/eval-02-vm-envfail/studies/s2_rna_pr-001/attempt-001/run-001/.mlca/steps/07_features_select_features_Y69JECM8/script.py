import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group = protocol["group_column"]
task = protocol["task"]

if len(targets) != 1:
    raise ValueError(f"Expected exactly one target, got {targets}")
target = targets[0]

forbidden = set(targets) | excluded
if group is not None:
    forbidden.add(group)

train = pd.read_parquet(inputs["train"])
if target not in train.columns:
    raise KeyError(f"Target {target!r} not found in train columns")

with open(inputs["06_preprocess_fit_preprocessing_Z6Z720YG/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

feature_cols = [c for c in train.columns if c not in forbidden]
n_rows = len(train)
kept = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_bool_dtype(s.dtype):
        kept.append(c)
    elif (
        pd.api.types.is_object_dtype(s.dtype)
        or pd.api.types.is_string_dtype(s.dtype)
        or pd.api.types.is_integer_dtype(s.dtype)
    ) and s.nunique(dropna=False) == n_rows:
        continue
    else:
        kept.append(c)
feature_cols = kept

if hasattr(preprocessor, "feature_names_in_"):
    expected = list(preprocessor.feature_names_in_)
    bad = set(expected) & forbidden
    if bad:
        raise ValueError(f"Preprocessor expects forbidden columns: {sorted(bad)}")
    feature_cols = expected

X = train.loc[:, feature_cols]
y = train[target]

try:
    n_features = len(preprocessor.get_feature_names_out())
except Exception:
    try:
        n_features = preprocessor.transform(X.head(1)).shape[1]
    except Exception:
        n_features = len(feature_cols)

k = min(20, n_features)
if k < 1:
    raise ValueError("No transformed features available for SelectKBest")

selector = Pipeline([
    ("preprocessor", preprocessor),
    ("selector", SelectKBest(score_func=mutual_info_classif, k=k)),
])
mlca_runtime.fit_record(selector, X, "train", y=y)
mlca_runtime.save_model(selector, "selector.pkl")

print(f"train shape: {train.shape}")
print(f"feature columns: {len(feature_cols)}")
print(f"SelectKBest k: {k}")
