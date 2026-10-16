import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise ValueError(f"Missing protocol keys: {missing}. Available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]
task = protocol["task"]

train = pd.read_parquet(inputs["train"])

drop_cols = set(targets) | set(excluded) | {group}
feature_cols = []
for col in train.columns:
    if col in drop_cols:
        continue
    if pd.api.types.is_object_dtype(train[col]) or pd.api.types.is_string_dtype(train[col]):
        if train[col].nunique() == len(train):
            continue
    elif pd.api.types.is_integer_dtype(train[col]):
        if train[col].nunique() == len(train):
            continue
    feature_cols.append(col)

with open(inputs["05_preprocess_fit_preprocessing_XYF9R0M6/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, "feature_names_in_"):
    expected = list(preprocessor.feature_names_in_)
    if set(expected) != set(feature_cols):
        raise ValueError(f"Preprocessor expects {expected}, but feature_cols are {feature_cols}")
    feature_cols = expected

X_train = train[feature_cols].copy()
X_trans = preprocessor.transform(X_train)

if not isinstance(X_trans, pd.DataFrame):
    if hasattr(preprocessor, "get_feature_names_out"):
        cols = preprocessor.get_feature_names_out()
    else:
        cols = [f"f{i}" for i in range(X_trans.shape[1])]
    X_trans = pd.DataFrame(X_trans, columns=cols, index=X_train.index)

selector = Pipeline([("selector", "passthrough")])
mlca_runtime.fit_record(selector, X_trans, "train")
mlca_runtime.save_model(selector, "selector_all.pkl")

print(f"Train rows: {len(train)}")
print(f"Feature columns: {len(feature_cols)}")
print(f"Transformed shape: {X_trans.shape}")
print("Saved selector_all.pkl")
