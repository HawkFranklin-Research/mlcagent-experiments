import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.base import clone

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
protocol = json.loads(inp["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing '{key}'. Available top-level keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")

with open(inp["06_preprocess_fit_preprocessing_Q5T3SXM3/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)


def is_key_like(s: pd.Series) -> bool:
    if s.isna().any():
        return False
    if pd.api.types.is_integer_dtype(s):
        return s.is_unique and int(s.min()) >= 0
    if s.dtype == object:
        return s.is_unique
    return False


exclude = set(targets) | set(excluded)
if group:
    exclude.add(group)
feature_cols = [c for c in train.columns if c not in exclude and not is_key_like(train[c])]
X_train = train[feature_cols]
y_train = train[targets[0]]

steps = list(preprocessor.steps) if isinstance(preprocessor, Pipeline) else [("preprocessor", preprocessor)]
pipeline = Pipeline(steps + [("selector", SelectKBest(score_func=mutual_info_classif, k=20))])

temp = mlca_runtime.fit_record(clone(preprocessor), X_train, "train", y=y_train)
X_trans = temp.transform(X_train)
n_features = X_trans.shape[1] if hasattr(X_trans, "shape") else len(X_trans[0])
k = min(20, n_features)
pipeline.set_params(selector__k=k)
print(f"features_in={len(feature_cols)} features_out={n_features} k={k}")

mlca_runtime.fit_record(pipeline, X_train, "train", y=y_train)
mlca_runtime.save_model(pipeline, "selector.pkl")
print("saved selector.pkl")
