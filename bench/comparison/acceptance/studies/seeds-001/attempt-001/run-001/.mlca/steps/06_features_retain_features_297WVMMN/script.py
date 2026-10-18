import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def need(mapping, key):
    if key not in mapping:
        raise KeyError(f"Missing _protocol[{key!r}]. Available top-level keys: {sorted(mapping)}")
    return mapping[key]


mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()

protocol = json.loads(paths["_protocol"].read_text())
targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group = need(protocol, "group_column")
_ = need(protocol, "task")

train = pd.read_parquet(paths["train"])

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

feature_cols = []
for col in train.columns:
    if col in drop:
        continue
    s = train[col]
    is_text = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)
    is_int = pd.api.types.is_integer_dtype(s)
    if (is_text or is_int) and s.is_unique:
        continue
    feature_cols.append(col)

with open(paths["05_preprocess_fit_preprocessing_A5A2YZJ3/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "set_output"):
    try:
        pre.set_output(transform="pandas")
    except Exception:
        pass

input_cols = list(getattr(pre, "feature_names_in_", feature_cols))
bad = [c for c in input_cols if c not in train.columns or c in drop]
if bad:
    raise ValueError(f"Invalid preprocessor input columns: {bad}")

selector = FunctionTransformer(func=None, validate=False, feature_names_out="one-to-one")
if isinstance(pre, Pipeline):
    model = Pipeline(list(pre.steps) + [("passthrough_selector", selector)])
else:
    model = Pipeline([("preprocessor", pre), ("passthrough_selector", selector)])

X = train.loc[:, input_cols]
fitted = mlca_runtime.fit_record(model, X, partition="train")
mlca_runtime.save_model(fitted, "selector_all.pkl")

print(f"features: {len(input_cols)}")
print(f"steps: {list(fitted.named_steps)}")
print("saved selector_all.pkl")
