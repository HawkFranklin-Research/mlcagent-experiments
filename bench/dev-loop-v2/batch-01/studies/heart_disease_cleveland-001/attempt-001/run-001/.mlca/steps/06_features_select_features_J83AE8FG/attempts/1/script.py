import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_7578M7XP/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feature_cols = [c for c in train.columns if c not in drop]

input_cols = getattr(pre, "feature_names_in_", None)
if input_cols is not None:
    input_cols = list(input_cols)
    X_train = train[input_cols].copy()
else:
    input_cols = feature_cols
    X_train = train[feature_cols].copy()

target = targets[0]
y_train = train[target]

try:
    n_out = len(pre.get_feature_names_out())
except Exception:
    n_out = pre.transform(X_train.iloc[:5]).shape[1]
k = min(20, n_out)

pipe = Pipeline([
    ("pre", pre),
    ("select", SelectKBest(score_func=mutual_info_classif, k=k)),
])
mlca_runtime.fit_record(pipe, X_train, "train", y_train)
mlca_runtime.save_model(pipe, "selector.pkl")

print(f"train rows: {len(train)}, feature cols: {input_cols.__len__()}")
print(f"preprocessed width: {n_out}, k: {k}")
print(f"selected width: {pipe.named_steps['select'].get_support().sum()}")
