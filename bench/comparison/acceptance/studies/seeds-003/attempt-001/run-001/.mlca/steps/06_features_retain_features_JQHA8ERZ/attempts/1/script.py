import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()

proto = json.loads(paths["_protocol"].read_text())
available = sorted(proto)
for key in ("targets", "excluded_columns", "task"):
    if key not in proto:
        raise KeyError(f"Missing _protocol[{key!r}]; available top-level keys: {available}")

targets = proto["targets"] or []
excluded = proto["excluded_columns"] or []
group_col = proto.get("group_column")
forbidden = set(targets) | set(excluded)
if group_col:
    forbidden.add(group_col)

train = pd.read_parquet(paths["train"])

def key_like(s):
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return bool(s.notna().all() and s.nunique(dropna=False) == len(s))
    return False

computed_features = [c for c in train.columns if c not in forbidden and not key_like(train[c])]

with open(paths["05_preprocess_fit_preprocessing_0JVWWBDZ/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    feature_cols = list(pre.feature_names_in_)
    invalid = [c for c in feature_cols if c not in train.columns or c in forbidden or key_like(train[c])]
    if invalid:
        raise ValueError(f"Preprocessor expects invalid feature columns: {invalid}")
else:
    feature_cols = computed_features

if not feature_cols:
    raise ValueError("No feature columns available")

X_train = train.loc[:, feature_cols].copy()

pipe = Pipeline([
    ("preprocess", pre),
    ("selector", FunctionTransformer()),
])
pipe.set_output(transform="pandas")

mlca_runtime.fit_record(pipe, X_train, partition="train")
mlca_runtime.save_model(pipe, "selector_all.pkl")

print(f"train rows: {len(train)}")
print(f"feature columns: {len(feature_cols)}")
print("saved selector_all.pkl")
