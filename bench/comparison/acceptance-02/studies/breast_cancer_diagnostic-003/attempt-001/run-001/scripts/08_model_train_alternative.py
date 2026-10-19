import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

seed = 42
mlca_runtime.set_seed(seed)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])

with open(inputs["06_features_select_features_HGTRMBKM/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

if isinstance(selector, Pipeline) and len(getattr(selector, "named_steps", {})) > 0:
    frontend = selector
    src = "selector-pipeline"
else:
    with open(inputs["05_preprocess_fit_preprocessing_AKNK2K5H/preprocessor.pkl"], "rb") as f:
        prep = pickle.load(f)
    frontend = Pipeline([("preprocessor", prep), ("selector", selector)])
    src = "prep+selector"

if hasattr(frontend, "feature_names_in_"):
    feature_cols = list(frontend.feature_names_in_)
else:
    excluded = set(protocol["targets"]) | set(protocol.get("excluded_columns") or [])
    if protocol.get("group_column"):
        excluded.add(protocol["group_column"])
    feature_cols = [c for c in train.columns if c not in excluded]

target = protocol["targets"][0]
X_train = train[feature_cols]
y_train = train[target]

_, counts = np.unique(y_train.astype(str), return_counts=True)
cw = "balanced" if counts.min() / counts.sum() < 0.10 else None

clf = HistGradientBoostingClassifier(random_state=seed, class_weight=cw)
full = Pipeline([("frontend", frontend), ("clf", clf)])
fitted = mlca_runtime.fit_record(full, X_train, "train", y=y_train)

mlca_runtime.save_model(fitted, "alternative_model.pkl")
print("frontend:", src, "n_features_in:", len(feature_cols))
print("clf: HistGradientBoostingClassifier class_weight=", cw)
print("saved alternative_model.pkl")
