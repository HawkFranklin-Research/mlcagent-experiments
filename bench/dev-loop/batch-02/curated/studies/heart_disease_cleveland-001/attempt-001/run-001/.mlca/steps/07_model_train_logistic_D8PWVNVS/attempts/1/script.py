import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inputs["05_preprocess_fit_preprocessing_XHDQ4AZ4/preprocessor.pkl"])
sel = load_pkl(inputs["06_features_select_features_ZYCM1WYT/selector.pkl"])

if isinstance(sel, Pipeline) and len(sel.steps) > 1:
    frontend = sel
    steps = list(sel.steps)
elif isinstance(pre, Pipeline) and len(pre.steps) > 1:
    frontend = pre
    steps = list(pre.steps)
else:
    frontend = Pipeline([("pre", pre), ("sel", sel)])
    steps = [("pre", pre), ("sel", sel)]

def get_feature_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        first = obj.steps[0][1]
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
    return None

features = get_feature_names(frontend)
if features is None:
    n = len(train)
    features = []
    for c in train.columns:
        if c in targets or c in excluded or (group is not None and c == group):
            continue
        s = train[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            if s.nunique() == n and n > 1:
                continue
        elif pd.api.types.is_integer_dtype(s):
            u = np.sort(s.dropna().unique())
            if len(u) == n and n > 1 and np.all(np.diff(u) == 1):
                continue
        features.append(c)

X_train = train[features].copy()
y_train = train[targets[0]].copy()

clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
pipeline = Pipeline(steps + [("clf", clf)])

mlca_runtime.fit_record(pipeline, X_train, "train", y_train)
mlca_runtime.save_model(pipeline, "model.pkl")

print(f"Features: {len(features)}")
print(f"Train shape: {X_train.shape}")
print("Saved model.pkl")
