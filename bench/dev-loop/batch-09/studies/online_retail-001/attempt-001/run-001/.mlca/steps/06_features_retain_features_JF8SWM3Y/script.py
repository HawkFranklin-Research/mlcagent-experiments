import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise ValueError(
        f"Missing protocol keys {missing}; available top-level keys: {list(protocol.keys())}"
    )

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]
_ = protocol["task"]

_ = json.loads(inputs["05_preprocess_fit_preprocessing_MW2B6JEQ/manifest.json"].read_text())
with open(inputs["05_preprocess_fit_preprocessing_MW2B6JEQ/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

pre_in = getattr(preprocessor, "feature_names_in_", None)
if pre_in is not None:
    bad = [c for c in pre_in if c in set(targets) | set(excluded) | {group}]
    if bad:
        raise ValueError(f"Preprocessor expects forbidden columns: {bad}")

train = pd.read_parquet(inputs["train"])
forbidden = set(targets) | set(excluded) | {group}

def key_like(s: pd.Series) -> bool:
    if (
        pd.api.types.is_object_dtype(s.dtype)
        or pd.api.types.is_string_dtype(s.dtype)
        or isinstance(s.dtype, pd.CategoricalDtype)
    ):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.nunique(dropna=False) == len(s)
    return False

feature_cols = [
    c for c in train.columns
    if c not in forbidden and not key_like(train[c])
]
if not feature_cols:
    raise ValueError("No feature columns remain after exclusions")

X = train[feature_cols].copy()

selector = FunctionTransformer(
    func=None,
    inverse_func=None,
    validate=False,
    accept_sparse=True,
    feature_names_out="one-to-one",
)
fitted_selector = mlca_runtime.fit_record(selector, X, "train")
mlca_runtime.save_model(fitted_selector, "selector_all.pkl")

print(f"train rows={len(X)} features={len(feature_cols)}")
print("saved selector_all.pkl")
