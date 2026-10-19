import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())


def need(d, key):
    if key not in d:
        raise KeyError(f"Missing {key!r}; available top-level keys: {list(d)}")
    return d[key]


targets = list(need(protocol, "targets"))
excluded = list(need(protocol, "excluded_columns"))
task = need(protocol, "task")
group_col = protocol.get("group_column")
time_cols = set(task.get("time_columns", [])) if isinstance(task, dict) else set()

train = pd.read_parquet(inputs["train"])
excluded_set = set(targets) | set(excluded)
if group_col:
    excluded_set.add(group_col)


def key_like(s):
    if len(s) == 0 or s.nunique(dropna=True) != len(s):
        return False
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s.dtype):
        return bool(s.is_monotonic_increasing or s.is_monotonic_decreasing)
    return False


feature_cols = [c for c in train.columns if c not in excluded_set]
feature_cols = [c for c in feature_cols if c in time_cols or not key_like(train[c])]

with open(inputs["05_preprocess_fit_preprocessing_R4WEQED5/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    pre_cols = list(pre.feature_names_in_)
    if set(pre_cols) != set(feature_cols):
        raise ValueError(
            f"preprocessor feature mismatch: extra={set(pre_cols) - set(feature_cols)}, "
            f"missing={set(feature_cols) - set(pre_cols)}"
        )
else:
    pre_cols = feature_cols

bad = set(pre_cols) & excluded_set
if bad:
    raise ValueError(f"excluded columns reached preprocessor: {sorted(bad)}")

pipe = Pipeline([
    ("preprocessor", pre),
    ("selector", "passthrough"),
])
pipe.set_output(transform="pandas")

fitted = mlca_runtime.fit_record(
    pipe,
    train[pre_cols],
    "train",
    y=train[targets[0]],
)
mlca_runtime.save_model(fitted, "selector_all.pkl")
print(f"rows={len(train)} input_features={len(pre_cols)} selector=passthrough")
