import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def get_key(d, k):
    if k not in d:
        raise KeyError(f"Missing key '{k}' in _protocol. Available: {list(d.keys())}")
    return d[k]

targets = get_key(protocol, "targets")
excluded = get_key(protocol, "excluded_columns")
group = get_key(protocol, "group_column")

train = pd.read_parquet(inputs["train"])

with open(inputs["05_preprocess_fit_preprocessing_WXDY6VPK/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_retain_features_C7QR720T/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)

if hasattr(preprocessor, "feature_names_in_"):
    raw_cols = list(preprocessor.feature_names_in_)
elif isinstance(selector, Pipeline) and hasattr(selector, "feature_names_in_"):
    raw_cols = list(selector.feature_names_in_)
else:
    raise ValueError("No feature_names_in_ found in saved preprocessor or selector.")

raw_cols = [c for c in raw_cols if c not in targets and c not in excluded and c != group]

X_train = train[raw_cols]
y_train = train[targets]

clf = MultiOutputClassifier(LogisticRegression(max_iter=2000))

selector_has_prep = False
if isinstance(selector, Pipeline) and hasattr(selector, "feature_names_in_") and hasattr(preprocessor, "feature_names_in_"):
    if np.array_equal(selector.feature_names_in_, preprocessor.feature_names_in_):
        selector_has_prep = True

if selector_has_prep:
    pipeline = Pipeline([("selector", selector), ("clf", clf)])
else:
    pipeline = Pipeline([("preprocessor", preprocessor), ("selector", selector), ("clf", clf)])

model = mlca_runtime.fit_record(pipeline, X_train, "train", y=y_train)
mlca_runtime.save_model(model, "model.pkl")

print(f"Train shape: {X_train.shape}, targets: {list(y_train.columns)}")
print("Saved model.pkl")
