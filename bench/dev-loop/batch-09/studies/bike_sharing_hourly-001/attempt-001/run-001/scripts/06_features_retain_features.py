import json, pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing {k} in protocol. Available keys: {list(protocol.keys())}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]
train = pd.read_parquet(inputs["train"])

key_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = train[c]
    if pd.api.types.is_float_dtype(s.dtype):
        continue
    if s.nunique(dropna=False) == len(s):
        key_cols.append(c)

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col and c not in key_cols]

with open(inputs["05_preprocess_fit_preprocessing_Q23GTBD5/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, "feature_names_in_"):
    expected = list(preprocessor.feature_names_in_)
    for c in targets + excluded:
        if c in expected:
            raise ValueError(f"Target/excluded column {c} is in preprocessor input")
    missing = set(expected) - set(train.columns)
    if missing:
        raise ValueError(f"Missing columns required by preprocessor: {missing}")
    feature_cols = expected

if hasattr(preprocessor, "set_output"):
    try:
        preprocessor.set_output(transform="pandas")
    except Exception:
        pass

X = train.loc[:, feature_cols]
y = train[targets[0]]

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("selector", SelectKBest(k="all")),
])
try:
    pipeline.set_output(transform="pandas")
except Exception:
    pass

fitted = mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(fitted, "selector_all.pkl")
mlca_runtime.record_metric("n_features", len(feature_cols), "train")
mlca_runtime.record_metric("n_train_rows", len(X), "train")
print(f"train rows: {len(X)}")
print(f"feature columns: {len(feature_cols)}")
print("saved selector_all.pkl")
