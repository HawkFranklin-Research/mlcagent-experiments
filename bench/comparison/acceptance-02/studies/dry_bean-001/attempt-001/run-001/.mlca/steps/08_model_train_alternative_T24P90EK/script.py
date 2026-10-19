import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

seed = 42
mlca_runtime.set_seed(seed)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    if pd.api.types.is_integer_dtype(train[c]):
        if train[c].nunique() == len(train):
            continue
    feature_cols.append(c)

# Build on the most complete saved pipeline
with open(inputs["06_features_select_features_4X00PG4B/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

with open(inputs["05_preprocess_fit_preprocessing_RMJEH56H/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

X = train[feature_cols]
y = train[targets[0]]

model = HistGradientBoostingClassifier(random_state=seed)
m = mlca_runtime.fit_record(model, X, "train", y=y)
print("trained", len(X), "rows", len(feature_cols), "features")
mlca_runtime.save_model(m, "alternative_model.pkl")
