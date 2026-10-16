import mlca_runtime

import json
import pickle
import pandas as pd
from functools import partial
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
target = protocol["task"]["targets"][0]
excluded = set(protocol.get("excluded_columns", []) or [])

train = pd.read_parquet(inputs["train"])

with open(inputs["05_preprocess_fit_preprocessing_B920SHZP/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

feature_columns = [c for c in train.columns if c != target and c not in excluded]
X_train = train[feature_columns]
y_train = train[target]

n_features = preprocessor.transform(X_train).shape[1]
k = min(20, n_features)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("selector", SelectKBest(partial(mutual_info_classif, random_state=42), k=k)),
])

mlca_runtime.fit_record(pipeline, X_train, "train", y=y_train)
mlca_runtime.save_model(pipeline, "selector.pkl")

print(f"train {X_train.shape[0]}x{X_train.shape[1]}")
print(f"features {n_features}")
print(f"k {k}")
print("saved selector.pkl")
