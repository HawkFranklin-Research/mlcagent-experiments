import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out_dir = mlca_runtime.output_dir()

# Read protocol
protocol = json.loads(inputs["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys: {missing}. Available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]
task = protocol["task"]

# Load training data
train = pd.read_parquet(inputs["train"])

# Define feature columns once (RL-0007)
all_cols = list(train.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]
# Remove key-like columns: text codes or integer sequences whose values are all distinct and carry no measurement.
# Continuous measurements are never identifiers. We'll check for object/string columns with all distinct values,
# and integer columns with all distinct values that are not continuous (e.g., not float).
key_like = []
for c in feature_cols:
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=False) == len(s):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=False) == len(s):
            # Check if it's a continuous measurement? Integer sequences are key-like.
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

X = train[feature_cols].copy()
y = train[targets].copy() if targets else None

# Load preprocessor
with open(inputs["05_preprocess_fit_preprocessing_Z1YWJ829/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

# Inspect preprocessor (RL-0011)
if isinstance(preprocessor, Pipeline):
    print("Preprocessor is Pipeline with steps:", list(preprocessor.named_steps.keys()))
else:
    print("Preprocessor type:", type(preprocessor).__name__)

# Transform training data using preprocessor
X_trans = preprocessor.transform(X)
# If output is sparse, convert to dense? FunctionTransformer can handle sparse? It passes through.
# But we need to preserve feature names. If preprocessor outputs a DataFrame, we can get feature names.
if hasattr(preprocessor, "get_feature_names_out"):
    try:
        feature_names = preprocessor.get_feature_names_out()
        if not isinstance(X_trans, pd.DataFrame):
            X_trans = pd.DataFrame(X_trans, columns=feature_names, index=X.index)
    except Exception:
        pass

# Fit passthrough selector (identity transformer)
selector = FunctionTransformer()
mlca_runtime.fit_record(selector, X_trans, partition="train", y=None)

# Save selector
mlca_runtime.save_model(selector, "selector_all.pkl")

# Record facts
mlca_runtime.record_fact_candidate("n_features", len(feature_cols), source="selector_all.pkl")
mlca_runtime.record_fact_candidate("n_train_rows", len(X), source="selector_all.pkl")
print(f"Features: {len(feature_cols)}, Train rows: {len(X)}")
print("Saved selector_all.pkl")
