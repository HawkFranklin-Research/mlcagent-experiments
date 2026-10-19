import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())
_ = json.loads(inputs["05_preprocess_fit_preprocessing_KQAA3HGR/manifest.json"].read_text())
with open(inputs["05_preprocess_fit_preprocessing_KQAA3HGR/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

def req(d, key):
    if key not in d:
        raise KeyError(f"Missing {key!r}; available top-level keys: {list(d.keys())}")
    return d[key]

targets = list(req(protocol, "targets"))
excluded = list(req(protocol, "excluded_columns"))
group = req(protocol, "group_column")
_ = req(protocol, "task")

def is_key_like(s: pd.Series) -> bool:
    n = len(s)
    if n == 0 or s.isna().any():
        return False
    if pd.api.types.is_bool_dtype(s):
        return False
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == n
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == n
    return False

forbidden = set(targets) | set(excluded) | ({group} if group else set())
if hasattr(preprocessor, "feature_names_in_"):
    feature_columns = list(preprocessor.feature_names_in_)
else:
    feature_columns = [
        c for c in train.columns
        if c not in forbidden and not is_key_like(train[c])
    ]

bad = set(feature_columns) & forbidden
if bad:
    raise ValueError(f"Feature columns include forbidden columns: {bad}")
key_like = [c for c in feature_columns if is_key_like(train[c])]
if key_like:
    raise ValueError(f"Feature columns include key-like columns: {key_like}")

X_train = train[feature_columns].copy()

try:
    n_out = len(preprocessor.get_feature_names_out())
    if n_out <= 300 and hasattr(preprocessor, "set_output"):
        preprocessor.set_output(transform="pandas")
except Exception:
    pass

selector_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("selector", "passthrough"),
])
fitted = mlca_runtime.fit_record(selector_pipeline, X_train, partition="train")
selector_all = fitted if fitted is not None else selector_pipeline

mlca_runtime.save_model(selector_all, "selector_all.pkl")
mlca_runtime.record_metric("n_train_rows", float(len(X_train)), "train")
mlca_runtime.record_metric("n_features", float(len(feature_columns)), "train")

print(f"train rows: {len(train)}")
print(f"selector input columns: {len(feature_columns)}")
print("saved selector_all.pkl")
