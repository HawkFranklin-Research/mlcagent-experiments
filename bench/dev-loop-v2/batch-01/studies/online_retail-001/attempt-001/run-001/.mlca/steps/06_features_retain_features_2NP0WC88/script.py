import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(
        f"Missing protocol keys {missing}; available top-level keys: {sorted(protocol)}"
    )

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]
train = pd.read_parquet(inputs["train"])

def is_key_like(series):
    nonnull = series.dropna()
    if len(nonnull) == 0 or nonnull.nunique() != len(nonnull):
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return pd.api.types.is_integer_dtype(series)
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = nonnull.head(2000)
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.9:
                return False
        except Exception:
            pass
        return True
    return False

skip = set(targets) | set(excluded)
if group is not None:
    skip.add(group)

feature_cols = [
    c for c in train.columns
    if c not in skip and not is_key_like(train[c])
]
if not feature_cols:
    raise ValueError("No feature columns remain after exclusions and key-like filtering")

X = train[feature_cols].copy()

preproc_path = inputs["05_preprocess_fit_preprocessing_8WCJT1S7/preprocessor.pkl"]
with open(preproc_path, "rb") as f:
    preproc = pickle.load(f)

if hasattr(preproc, "feature_names_in_"):
    pcols = list(preproc.feature_names_in_)
    bad = [c for c in pcols if c in skip]
    if bad:
        raise ValueError(f"Preprocessor expects excluded columns: {bad}")
    feature_cols = pcols
    X = train[feature_cols].copy()

if hasattr(preproc, "set_output"):
    try:
        preproc.set_output(transform="pandas")
    except Exception:
        pass

selector = ColumnTransformer(
    transformers=[],
    remainder="passthrough",
    verbose_feature_names_out=False,
    sparse_threshold=0.0,
)
try:
    selector.set_output(transform="pandas")
except Exception:
    pass

if isinstance(preproc, Pipeline):
    model = Pipeline(list(preproc.steps) + [("selector_all", selector)])
else:
    model = Pipeline([
        ("preprocessor", preproc),
        ("selector_all", selector),
    ])

mlca_runtime.fit_record(model, X, partition="train")
mlca_runtime.save_model(model, "selector_all.pkl")
mlca_runtime.record_metric("n_features_in", X.shape[1], "train")
print(f"Fitted selector_all on {X.shape[0]} rows and {X.shape[1]} features")
