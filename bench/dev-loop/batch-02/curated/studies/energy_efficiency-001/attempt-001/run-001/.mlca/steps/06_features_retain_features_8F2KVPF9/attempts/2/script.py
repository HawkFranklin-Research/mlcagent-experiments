import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.feature_selection import VarianceThreshold

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing {key} in _protocol. Available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]

train = pd.read_parquet(inputs["train"])

all_cols = list(train.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group]

key_like = []
for c in feature_cols:
    s = train[c]
    if s.dtype == 'object' or pd.api.types.is_string_dtype(s):
        if s.nunique() == len(s):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique() == len(s):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

X = train[feature_cols]

with open(inputs["05_preprocess_fit_preprocessing_K5VFSMR2/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

X_trans = preprocessor.transform(X)
if hasattr(X_trans, "toarray"):
    X_trans = X_trans.toarray()
if not isinstance(X_trans, pd.DataFrame):
    if hasattr(preprocessor, "get_feature_names_out"):
        names = preprocessor.get_feature_names_out()
    else:
        names = [f"f{i}" for i in range(X_trans.shape[1])]
    X_trans = pd.DataFrame(X_trans, columns=names, index=X.index)

selector = VarianceThreshold(threshold=0.0)
mlca_runtime.fit_record(selector, X_trans, partition="train")

mlca_runtime.save_model(selector, "selector_all.pkl")

print(f"Features: {X_trans.shape[1]}, selected: {selector.get_support().sum()}")
