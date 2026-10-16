import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

with open(inputs["05_preprocess_fit_preprocessing_S3GJVMHP/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

with open(inputs["05_preprocess_fit_preprocessing_S3GJVMHP/manifest.json"], "r") as f:
    manifest = json.load(f)

if "columns" in protocol and "target" in protocol["columns"]:
    target = protocol["columns"]["target"]
elif "task" in protocol and "targets" in protocol["task"] and protocol["task"]["targets"]:
    target = protocol["task"]["targets"][0]
else:
    target = "heart_disease_binary"

excluded_columns = protocol.get("excluded_columns", ["num"])

feature_cols = [c for c in train.columns if c != target and c not in excluded_columns]

X_train = train[feature_cols]
y_train = train[target]
X_val = val[feature_cols]
y_val = val[target]

n_features = X_train.shape[1]
k = min(20, n_features)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("selector", SelectKBest(mutual_info_classif, k=k)),
])

mlca_runtime.fit_record(pipeline, X_train, "train", y=y_train)

mlca_runtime.save_model(pipeline, "selector.pkl")

selector = pipeline.named_steps["selector"]
n_selected = int(selector.get_support().sum())

mlca_runtime.record_metric("n_selected_features", float(n_selected), "train")

print(f"train shape: {X_train.shape}")
print(f"val shape: {X_val.shape}")
print(f"n_features: {n_features}")
print(f"k: {k}")
print(f"n_selected: {n_selected}")
