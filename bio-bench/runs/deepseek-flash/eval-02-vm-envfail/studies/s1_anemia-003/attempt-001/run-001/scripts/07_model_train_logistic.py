import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
excluded = list(protocol.get("excluded_columns") or [])
group = protocol.get("group_column")

train = pd.read_parquet(inp["train"])
with open(inp["05_preprocess_fit_preprocessing_T6A9FCB5/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inp["06_features_retain_features_WE31FN6Z/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feat_cols = [c for c in train.columns if c not in drop]

n = len(train)
key_like = []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        u = np.sort(s.dropna().unique())
        if len(u) == n and ((u[0] == 0 and u[-1] == n - 1) or (u[0] == 1 and u[-1] == n)):
            key_like.append(c)
    elif pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
        if s.nunique(dropna=True) == n and n > 1:
            key_like.append(c)
feat_cols = [c for c in feat_cols if c not in key_like]

clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
sel_in = list(selector.feature_names_in_) if hasattr(selector, "feature_names_in_") else None
prep_steps = list(preprocessor.named_steps.keys()) if isinstance(preprocessor, Pipeline) else []

if isinstance(selector, Pipeline) and sel_in is not None and set(sel_in) == set(feat_cols):
    pipe = Pipeline([("sel", selector), ("clf", clf)])
    frontend = selector
elif any(k in prep_steps for k in ["sel", "selector", "feature_select"]):
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    frontend = preprocessor
else:
    pipe = Pipeline([("prep", preprocessor), ("sel", selector), ("clf", clf)])
    frontend = preprocessor

use_cols = list(frontend.feature_names_in_) if hasattr(frontend, "feature_names_in_") else feat_cols
X = train[use_cols].copy()
y = train[targets[0]].copy()

mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(pipe, "model.pkl")
print(f"rows={len(X)} cols={X.shape[1]} saved=model.pkl")
