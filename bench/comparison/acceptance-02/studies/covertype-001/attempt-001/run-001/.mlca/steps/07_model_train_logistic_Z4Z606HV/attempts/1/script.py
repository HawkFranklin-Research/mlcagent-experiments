import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

train = pd.read_parquet(inputs["train"])
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
target = targets[0]
feature_cols = [c for c in train.columns if c not in targets and c not in excluded]

with open(inputs["05_preprocess_fit_preprocessing_Z8NMCAXW/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_53HVANE8/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

pre_in = list(getattr(pre, "feature_names_in_", []))
sel_in = list(getattr(sel, "feature_names_in_", []))
if pre_in and sel_in and pre_in == sel_in:
    raise RuntimeError("preprocessor and selector both expect identical raw columns; cannot chain (RL-0027).")

pipeline = Pipeline([
    ("preprocess", pre),
    ("select", sel),
    ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
])

X = train[feature_cols]
y = train[target]
mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(pipeline, "model.pkl")
print(f"fit rows={len(X)} feats={len(feature_cols)} classes={y.nunique()}")
