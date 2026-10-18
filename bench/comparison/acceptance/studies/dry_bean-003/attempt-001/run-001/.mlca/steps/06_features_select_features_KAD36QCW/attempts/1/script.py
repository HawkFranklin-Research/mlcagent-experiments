import json
import pickle

import pandas as pd
import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
from pandas.api.types import is_object_dtype, is_string_dtype, is_integer_dtype

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing {key}; available top-level keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")

train = pd.read_parquet(paths["train"])
if not targets or targets[0] not in train.columns:
    raise KeyError(f"Target column not found; available columns: {list(train.columns)}")
y = train[targets[0]]

bad = set(targets) | set(excluded)
if group is not None:
    bad.add(group)

n = len(train)
key_like = []
for col in train.columns:
    if col in bad:
        continue
    s = train[col]
    if is_object_dtype(s.dtype) or is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=False) == n:
            key_like.append(col)
    elif is_integer_dtype(s.dtype):
        if s.nunique(dropna=False) == n:
            key_like.append(col)

feature_cols = [c for c in train.columns if c not in bad and c not in set(key_like)]
if not feature_cols:
    raise ValueError("No feature columns remain after exclusions")

with open(paths["05_preprocess_fit_preprocessing_DYJSGEW8/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if isinstance(pre, Pipeline):
    _ = pre.named_steps

pre_cols = getattr(pre, "feature_names_in_", None)
if pre_cols is not None:
    pre_cols = list(pre_cols)
    if any(c in bad for c in pre_cols):
        raise ValueError("Preprocessor expects a target/excluded/group column")
    if set(pre_cols) != set(feature_cols):
        if set(pre_cols).issubset(set(feature_cols)):
            feature_cols = pre_cols
        else:
            raise ValueError("Preprocessor input columns do not match feature columns")
    else:
        feature_cols = pre_cols

X = train[feature_cols]

try:
    Xt = pre.transform(X)
except NotFittedError:
    mlca_runtime.fit_record(pre, X, "train", y=y)
    Xt = pre.transform(X)

n_features = Xt.shape[1] if hasattr(Xt, "shape") else len(pre.get_feature_names_out())
k = min(20, int(n_features))
selector = SelectKBest(score_func=mutual_info_classif, k=k)
pipeline = Pipeline([("preprocess", pre), ("selector", selector)])
mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(pipeline, "selector.pkl")

mlca_runtime.record_metric("n_feature_columns", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_transformed_features", float(n_features), "train")
mlca_runtime.record_metric("k_selected", float(k), "train")

print(f"train shape: {train.shape}")
print(f"feature columns: {len(feature_cols)}; transformed: {n_features}; k: {k}")
