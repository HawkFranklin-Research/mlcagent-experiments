import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest

mlca_runtime.set_seed(42)

inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
train = pd.read_parquet(inp["train"])

def required(protocol, key):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {list(protocol)}")
    return protocol[key]

targets = required(protocol, "targets")
excluded = required(protocol, "excluded_columns")
group = protocol.get("group_column")
target = targets[0]
blocked = set(targets) | set(excluded) | ({group} if group else set())

def key_like(s):
    if pd.api.types.is_float_dtype(s.dtype):
        return False
    if pd.api.types.is_integer_dtype(s.dtype) or pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        return len(s) > 0 and s.nunique(dropna=True) == len(s)
    return False

computed_features = [
    c for c in train.columns
    if c not in blocked and not key_like(train[c])
]

with open(inp["05_preprocess_fit_preprocessing_FMADE2YX/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

feature_cols = computed_features
if hasattr(pre, "feature_names_in_"):
    pre_cols = list(pre.feature_names_in_)
    if set(pre_cols) != set(computed_features):
        raise ValueError(
            f"Preprocessor inputs differ from computed feature columns: "
            f"{pre_cols} vs {computed_features}"
        )
    feature_cols = pre_cols

if not feature_cols:
    raise ValueError("No feature columns remain after exclusions and key-like filtering")

X = train[feature_cols].copy()
y = train[target].copy()

pipe = Pipeline([
    ("preprocessor", pre),
    ("selector", SelectKBest(k="all")),
])
try:
    pipe.set_output(transform="pandas")
except Exception:
    pass

fitted = mlca_runtime.fit_record(pipe, X, partition="train", y=y)
mlca_runtime.save_model(fitted, "selector_all.pkl")

n_selected = int(fitted.named_steps["selector"].get_support().sum())
mlca_runtime.record_metric("n_raw_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_selected_features", float(n_selected), "train")
print(f"train rows: {len(train)}")
print(f"raw feature columns: {len(feature_cols)}")
print(f"selected features: {n_selected}")
