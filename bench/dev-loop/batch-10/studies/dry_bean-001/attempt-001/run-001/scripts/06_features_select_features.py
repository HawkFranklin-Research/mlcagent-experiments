import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = list(protocol.get("excluded_columns", []) or [])
group = protocol.get("group_column")
task = protocol["task"]

train = pd.read_parquet(inputs["train"])

with open(inputs["05_preprocess_fit_preprocessing_G69EDXSB/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, "feature_names_in_"):
    feature_cols = [c for c in preprocessor.feature_names_in_ if c in train.columns]
else:
    drop = set(targets) | set(excluded) | ({group} if group else set())
    feature_cols = [c for c in train.columns if c not in drop]

X = train[feature_cols]
y = train[targets[0]]

X_pre = preprocessor.transform(X)
n_features = int(X_pre.shape[1]) if hasattr(X_pre, "shape") else len(X_pre[0])
k = min(20, n_features)

selector = SelectKBest(score_func=mutual_info_classif, k=k)
mlca_runtime.fit_record(selector, X_pre, partition="train", y=y)

pipe = Pipeline([("pre", preprocessor), ("select", selector)])
mlca_runtime.save_model(pipe, "selector.pkl")

mlca_runtime.record_metric("n_features_in", float(n_features), "train")
mlca_runtime.record_metric("n_features_selected", float(k), "train")
print(f"features_in={n_features} selected={k}")
