import json
import pickle

import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"] or [])
group_col = protocol["group_column"]

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_Y0BA94GV/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)


def is_key_like(s: pd.Series) -> bool:
    n = len(s)
    if n == 0:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == n
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == n
    return False


feature_cols = []
for col in train.columns:
    if col in targets or col in excluded or (group_col is not None and col == group_col):
        continue
    if is_key_like(train[col]):
        continue
    feature_cols.append(col)

if hasattr(pre, "feature_names_in_"):
    input_cols = list(pre.feature_names_in_)
else:
    input_cols = feature_cols

for col in input_cols:
    if col not in train.columns:
        raise KeyError(f"Preprocessor input column {col!r} not found in train")
    if col in targets or col in excluded or (group_col is not None and col == group_col):
        raise ValueError(f"Target or excluded column {col!r} found in preprocessor inputs")

X = train.loc[:, input_cols]
y = train.loc[:, targets]

selector = FunctionTransformer(validate=False, feature_names_out="one-to-one")
pipeline = Pipeline([("preprocess", pre), ("selector", selector)])
fitted = mlca_runtime.fit_record(pipeline, X, "train", y)
mlca_runtime.save_model(fitted, "selector_all.pkl")

print(f"rows={len(X)} features={len(input_cols)}")
print("saved selector_all.pkl")
