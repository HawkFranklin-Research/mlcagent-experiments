import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
task = protocol["task"]

train = pd.read_parquet(inputs["train"])

with open(inputs["05_preprocess_fit_preprocessing_XHNDRHCS/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

feat_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
feat_cols = [c for c in feat_cols if c in set(pre.feature_names_in_)] if hasattr(pre, "feature_names_in_") else feat_cols

y = train[targets[0]]
X = train[feat_cols]

if isinstance(pre, Pipeline) and "selectkbest" in pre.named_steps:
    frontend = pre
    selector = frontend.named_steps["selectkbest"]
    selector.k = min(20, len(feat_cols))
else:
    selector = SelectKBest(score_func=mutual_info_classif, k=min(20, len(feat_cols)))
    if hasattr(pre, "feature_names_in_"):
        frontend = Pipeline([("pre", pre), ("selectkbest", selector)])
    else:
        frontend = Pipeline([("pre", pre), ("selectkbest", selector)])

mlca_runtime.fit_record(frontend, X, "train", y=y)
mlca_runtime.save_model(frontend, "selector.pkl")

n_feats = len(feat_cols)
print(f"features in: {n_feats}, k: {min(20, n_feats)}, targets: {targets}")
