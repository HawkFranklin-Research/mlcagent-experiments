import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def required(d, key):
    if key not in d:
        raise KeyError(f"Missing required key {key!r}; available top-level keys: {sorted(d.keys())}")
    return d[key]

targets = required(protocol, "targets")
excluded = required(protocol, "excluded_columns")
group = required(protocol, "group_column")
_ = required(protocol, "task")

train = pd.read_parquet(inputs["train"])
pre_path = inputs["05_preprocess_fit_preprocessing_CK5QBBYX/preprocessor.pkl"]
with open(pre_path, "rb") as f:
    pre = pickle.load(f)

def is_key_like(s):
    n = len(s)
    if s.nunique(dropna=False) != n:
        return False
    kind = s.dtype.kind
    if kind in "OSU":
        return True
    if kind in "iu":
        v = s.dropna()
        return len(v) == n and v.max() - v.min() + 1 == n
    return False

exclude = set(targets) | set(excluded)
if group is not None:
    exclude.add(group)

candidate = [c for c in train.columns if c not in exclude and not is_key_like(train[c])]
if hasattr(pre, "feature_names_in_"):
    feature_cols = [c for c in pre.feature_names_in_ if c in train.columns]
else:
    feature_cols = candidate

leak = set(feature_cols) & (set(targets) | set(excluded) | ({group} if group is not None else set()))
if leak:
    raise ValueError(f"Leakage columns in features: {sorted(leak)}")

X = train.loc[:, feature_cols].copy()
if not hasattr(pre, "transform"):
    raise TypeError("Loaded preprocessor has no transform method")

X_trans = pre.transform(X)
if X_trans.shape[1] <= 300 and hasattr(pre, "set_output"):
    pre.set_output(transform="pandas")
    X_trans = pre.transform(X)

selector = FunctionTransformer(validate=False)
mlca_runtime.fit_record(selector, X_trans, "train")

steps = list(pre.steps) if isinstance(pre, Pipeline) else [("preprocessor", pre)]
name = "selector"
used = {n for n, _ in steps}
while name in used:
    name += "_"
pipeline = Pipeline(steps=steps + [(name, selector)])

mlca_runtime.save_model(pipeline, "selector_all.pkl")
mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_transformed_features", float(X_trans.shape[1]), "train")

print(f"train rows: {len(train)}")
print(f"features: {len(feature_cols)}")
print(f"transformed width: {X_trans.shape[1]}")
print("saved selector_all.pkl")
