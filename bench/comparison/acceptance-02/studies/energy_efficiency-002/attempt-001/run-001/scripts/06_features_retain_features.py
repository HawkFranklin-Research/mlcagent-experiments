import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

try:
    targets = list(protocol["targets"])
    excluded = list(protocol["excluded_columns"])
    group_col = protocol["group_column"]
    task = protocol["task"]
except KeyError as exc:
    raise RuntimeError(
        f"Missing protocol key {exc}; available top-level keys: {list(protocol.keys())}"
    ) from exc

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_C8Q2SNQ9/preprocessor.pkl"], "rb") as fh:
    preprocessor = pickle.load(fh)
_ = json.loads(
    inputs["05_preprocess_fit_preprocessing_C8Q2SNQ9/manifest.json"].read_text()
)

exclude = set(targets) | set(excluded)
if group_col is not None:
    exclude.add(group_col)

feature_cols = []
for col in train.columns:
    if col in exclude:
        continue
    s = train[col]
    all_distinct = s.nunique(dropna=False) == len(s)
    is_text_code = (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
    )
    is_integer_sequence = pd.api.types.is_integer_dtype(s)
    if all_distinct and (is_text_code or is_integer_sequence):
        continue
    feature_cols.append(col)

if hasattr(preprocessor, "feature_names_in_"):
    expected = list(preprocessor.feature_names_in_)
    missing = [c for c in expected if c not in feature_cols]
    if missing:
        raise RuntimeError(f"Preprocessor expects non-feature columns: {missing}")
    feature_cols = expected

if hasattr(preprocessor, "set_output"):
    try:
        preprocessor.set_output(transform="pandas")
    except Exception:
        pass

selector = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("passthrough", "passthrough"),
    ]
)
if hasattr(selector, "set_output"):
    try:
        selector.set_output(transform="pandas")
    except Exception:
        pass

X = train[feature_cols].copy()
fitted = mlca_runtime.fit_record(selector, X, "train")
mlca_runtime.save_model(fitted, "selector_all.pkl")

print(f"train rows={len(train)}, features={len(feature_cols)}, targets={len(targets)}")
print("saved selector_all.pkl")
