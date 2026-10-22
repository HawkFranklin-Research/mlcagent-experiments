import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

def load_json(path):
    return json.loads(path.read_text())

protocol = load_json(inputs["_protocol"])
for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing {key!r} in _protocol; available top-level keys: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]
forbidden = set(targets) | set(excluded) | ({group} if group is not None else set())

train = pd.read_parquet(inputs["train"])
manifest = load_json(inputs["05_preprocess_fit_preprocessing_A4M5DTDE/manifest.json"])

def key_like(s):
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        name = str(s.name).lower()
        if any(t in name for t in ("id", "key", "code")) and s.nunique(dropna=False) == len(s):
            return True
    return False

computed = [c for c in train.columns if c not in forbidden and not key_like(train[c])]

with open(inputs["05_preprocess_fit_preprocessing_A4M5DTDE/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, "feature_names_in_"):
    prep_cols = list(preprocessor.feature_names_in_)
    if forbidden.intersection(prep_cols):
        raise ValueError("Preprocessor expects forbidden target/excluded/group columns")
    feature_cols = prep_cols
else:
    feature_cols = computed

X_raw = train[feature_cols]
Xt = preprocessor.transform(X_raw)
if hasattr(Xt, "shape") and Xt.shape[1] <= 300 and hasattr(preprocessor, "set_output"):
    try:
        preprocessor.set_output(transform="pandas")
        Xt = preprocessor.transform(X_raw)
    except (ValueError, TypeError):
        pass

selector = FunctionTransformer(validate=False, feature_names_out="one-to-one")
selector = mlca_runtime.fit_record(selector, Xt, partition="train")

if hasattr(Xt, "columns"):
    out_names = list(Xt.columns)
elif hasattr(preprocessor, "get_feature_names_out"):
    out_names = list(preprocessor.get_feature_names_out())
else:
    out_names = None

if out_names is not None:
    selector.feature_names_in_ = out_names
    selector.n_features_in_ = len(out_names)

full = Pipeline([("preprocessor", preprocessor), ("selector", selector)])
mlca_runtime.save_model(full, "selector_all.pkl")

print(f"train rows: {len(train)}")
print(f"features: {len(feature_cols)}")
print("saved selector_all.pkl")
