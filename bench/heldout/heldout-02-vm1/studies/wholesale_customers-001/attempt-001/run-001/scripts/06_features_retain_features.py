import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol)}")

with open(inputs["05_preprocess_fit_preprocessing_RXV573Z5/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

feature_cols = list(getattr(preprocessor, "feature_names_in_", []))
if not feature_cols:
    drop = set(protocol["targets"]) | set(protocol["excluded_columns"])
    group = protocol["group_column"]
    if group:
        drop.add(group)
    feature_cols = []
    for col in train.columns:
        if col in drop:
            continue
        s = train[col]
        if (pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)) and s.nunique(dropna=False) == len(s):
            continue
        if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=False) == len(s):
            vals = s.dropna().sort_values().unique()
            if len(vals) == len(s) and vals.max() - vals.min() + 1 == len(vals):
                continue
        feature_cols.append(col)

X_raw = train[feature_cols].copy()
if hasattr(preprocessor, "set_output"):
    try:
        preprocessor.set_output(transform="pandas")
    except (TypeError, ValueError):
        pass

X_proc = preprocessor.transform(X_raw)
if not isinstance(X_proc, pd.DataFrame) and X_proc.shape[1] <= 300:
    names = getattr(preprocessor, "get_feature_names_out", lambda: None)()
    if names is not None:
        X_proc = pd.DataFrame(X_proc, columns=list(names), index=X_raw.index)

selector = FunctionTransformer(func=None, validate=False, accept_sparse=True)
fitted = mlca_runtime.fit_record(selector, X_proc, "train")
mlca_runtime.save_model(fitted, "selector_all.pkl")

print(f"train_rows={len(train)}")
print(f"selector_input_features={getattr(fitted, 'n_features_in_', X_proc.shape[1])}")
