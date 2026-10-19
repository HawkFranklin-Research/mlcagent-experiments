import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from functools import partial
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
ip = mlca_runtime.inputs()
protocol = json.loads(ip["_protocol"].read_text())
if "targets" not in protocol:
    raise KeyError(f"_protocol missing 'targets'; available keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns") or [])
group = protocol.get("group_column", None)
if group is not None and group in targets:
    raise ValueError("group_column overlaps targets")

train = pd.read_parquet(ip["train"])
missing = [c for c in targets if c not in train.columns]
if missing:
    raise KeyError(f"target columns missing from train: {missing}; columns = {list(train.columns)}")
y = train[targets[0]]

# RL-0007 feature list: everything but targets, excluded, group, and key-like all-distinct codes/ints
feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or (group is not None and c == group):
        continue
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=False) == len(s):
            continue
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = s.dropna().to_numpy()
        if v.size and len(np.unique(v)) == len(v) == len(s) and (v.max() - v.min() + 1) == len(v):
            continue
    feature_cols.append(c)

with open(ip["05_preprocess_fit_preprocessing_RMJEH56H/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

# RL-0011: preserve the earlier pipeline's exact input columns and order
if getattr(pre, "feature_names_in_", None) is not None:
    names = [str(c) for c in pre.feature_names_in_]
    leaked = [c for c in names if c in targets or c in excluded or (group is not None and c == group)]
    if leaked:
        raise ValueError(f"preprocessor expects target/excluded columns: {leaked}")
    feature_cols = names

X_train = train[feature_cols]
Xt = pre.transform(X_train)
n_feat = int(Xt.shape[1]) if hasattr(Xt, "shape") else int(np.asarray(Xt).shape[1])
k = min(20, n_feat)

steps = list(pre.steps) if isinstance(pre, Pipeline) else [("preprocessor", pre)]
pipe = Pipeline(steps + [
    ("select", SelectKBest(score_func=partial(mutual_info_classif, random_state=42), k=k))
])

mlca_runtime.fit_record(pipe, X_train, "train", y=y)
mlca_runtime.save_model(pipe, "selector.pkl")
mlca_runtime.record_metric("n_input_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_selected_features", float(k), "train")
print(f"n_input={len(feature_cols)} n_selected={k} rows={len(X_train)}")
