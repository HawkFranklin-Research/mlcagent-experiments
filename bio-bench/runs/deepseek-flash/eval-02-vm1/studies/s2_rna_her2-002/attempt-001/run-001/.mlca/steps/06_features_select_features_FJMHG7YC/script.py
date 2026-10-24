import json
import pickle

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
if len(targets) != 1:
    raise ValueError(f"Expected exactly one target, got {targets}")
target = targets[0]

excluded_raw = protocol["excluded_columns"]
if not isinstance(excluded_raw, (list, tuple, set)):
    raise TypeError(f"excluded_columns must be a list/tuple/set, got {type(excluded_raw).__name__}")
excluded = set(excluded_raw)
group = protocol["group_column"]
if group is not None:
    excluded.add(group)

with inputs["05_preprocess_fit_preprocessing_R1GAM795/preprocessor.pkl"].open("rb") as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, "feature_names_in_"):
    feature_cols = list(preprocessor.feature_names_in_)
else:
    feature_cols = []
    for col in train.columns:
        if col in excluded or col == target:
            continue
        s = train[col]
        if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
            if s.nunique(dropna=True) == len(s):
                continue
        if pd.api.types.is_integer_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype):
            if s.nunique(dropna=True) == len(s) and s.is_monotonic_increasing:
                continue
        feature_cols.append(col)

bad = [c for c in feature_cols if c in excluded or c == target]
if bad:
    raise ValueError(f"Feature list contains excluded/target columns: {bad}")
missing_features = [c for c in feature_cols if c not in train.columns]
if missing_features:
    raise ValueError(f"Features missing from training table: {missing_features}")

X_train = train.loc[:, feature_cols].copy()
y_train = train[target]

try:
    n_features = len(preprocessor.get_feature_names_out())
except Exception:
    sample = X_train.head(min(10, len(X_train)))
    n_features = preprocessor.transform(sample).shape[1]

if n_features == 0:
    raise ValueError("Preprocessor produced zero features")

k = min(20, n_features)
selector_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("select", SelectKBest(score_func=mutual_info_classif, k=k)),
    ]
)

selector_pipeline = mlca_runtime.fit_record(selector_pipeline, X_train, "train", y_train)
mlca_runtime.save_model(selector_pipeline, "selector.pkl")

print(f"train rows={len(train)}")
print(f"raw features={len(feature_cols)}")
print(f"encoded features={n_features}, selected={k}")
print("saved selector.pkl")
