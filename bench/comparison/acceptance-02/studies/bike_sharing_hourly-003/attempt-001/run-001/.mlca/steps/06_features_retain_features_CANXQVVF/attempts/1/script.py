import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_SMPGAWB4/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
_ = json.loads(inputs["05_preprocess_fit_preprocessing_SMPGAWB4/manifest.json"].read_text())

for key in ("targets", "excluded_columns", "group_column"):
    if key not in protocol:
        raise KeyError(f"{key!r} missing; available top-level keys: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]
drop = set(targets) | set(excluded)
if group is not None:
    drop |= set(group) if isinstance(group, (list, tuple, set)) else {group}

n = len(train)
feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if s.nunique(dropna=False) == n and (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
        or pd.api.types.is_integer_dtype(s)
    ):
        continue
    feature_cols.append(c)

if hasattr(preprocessor, "feature_names_in_"):
    expected = list(preprocessor.feature_names_in_)
    if set(expected) != set(feature_cols):
        raise ValueError("Preprocessor feature columns do not match computed feature columns")
    feature_cols = expected

X_raw = train[feature_cols].copy()
try:
    preprocessor.set_output(transform="pandas")
except Exception:
    pass
X_pre = preprocessor.transform(X_raw)

if not isinstance(X_pre, pd.DataFrame) and hasattr(preprocessor, "get_feature_names_out"):
    names = preprocessor.get_feature_names_out()
    if len(names) <= 300:
        if hasattr(X_pre, "toarray"):
            X_pre = X_pre.toarray()
        X_pre = pd.DataFrame(X_pre, columns=names, index=X_raw.index)

selector = FunctionTransformer(validate=False)
fitted_selector = mlca_runtime.fit_record(selector, X_pre, "train")
mlca_runtime.save_model(fitted_selector, "selector_all.pkl")
print(f"rows: {len(X_raw)} raw_features: {len(feature_cols)} encoded_shape: {getattr(X_pre, 'shape', None)}")
