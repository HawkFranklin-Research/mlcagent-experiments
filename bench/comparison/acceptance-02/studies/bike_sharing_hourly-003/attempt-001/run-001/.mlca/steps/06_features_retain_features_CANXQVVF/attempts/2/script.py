import json
import pickle
import pandas as pd
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

if hasattr(preprocessor, "feature_names_in_"):
    feature_cols = list(preprocessor.feature_names_in_)
else:
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

present = [c for c in feature_cols if c in train.columns]
missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise ValueError(f"Expected feature columns missing from train: {missing[:5]}")

X_raw = train[present].copy()
selector = FunctionTransformer(validate=False)
fitted_selector = mlca_runtime.fit_record(selector, X_raw, "train")
mlca_runtime.save_model(fitted_selector, "selector_all.pkl")
mlca_runtime.record_fact_candidate("selector_features", int(X_raw.shape[1]), source="selector_all.pkl")
print(f"rows: {len(X_raw)} features: {X_raw.shape[1]}")
