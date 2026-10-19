import json
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
if "targets" not in proto:
    raise KeyError(f"Missing _protocol key 'targets'; available keys: {sorted(proto)}")

targets = list(proto["targets"])
excluded = list(proto.get("excluded_columns", []))
group = proto.get("group_column", None)
train = pd.read_parquet(inp["train"])
n = len(train)

def is_key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=False) == n
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == n and s.is_monotonic_increasing
    return False

candidate_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded
    and (group is None or c != group)
    and not is_key_like(train[c])
]

with open(inp["05_preprocess_fit_preprocessing_8APC2384/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_select_features_23ZRTGDS/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    return None

def output_names(obj):
    if hasattr(obj, "get_feature_names_out"):
        try:
            return list(obj.get_feature_names_out())
        except Exception:
            return None
    return None

pre_in = input_names(pre)
sel_in = input_names(sel)
pre_out = output_names(pre)
clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

if isinstance(sel, Pipeline) and len(sel.named_steps) > 1 and sel_in is not None and pre_in is not None and set(sel_in) == set(pre_in):
    steps = [("pipeline", sel), ("classifier", clf)]
elif isinstance(sel, Pipeline) and len(sel.named_steps) > 1:
    steps = [("pipeline", sel), ("classifier", clf)]
elif sel_in is not None and pre_out is not None and set(sel_in) == set(pre_out):
    steps = [("preprocessing", pre), ("selector", sel), ("classifier", clf)]
elif sel_in is not None and pre_in is not None and set(sel_in) == set(pre_in) and not isinstance(sel, Pipeline):
    steps = [("preprocessing", pre), ("classifier", clf)]
else:
    steps = [("preprocessing", pre), ("selector", sel), ("classifier", clf)]

front = steps[0][1]
feature_cols = input_names(front)
if feature_cols is None:
    feature_cols = candidate_cols

banned = set(targets) | set(excluded)
if group is not None:
    banned.add(group)
bad = set(feature_cols) & banned
if bad:
    raise ValueError(f"Features include banned columns: {sorted(bad)}")
missing = set(feature_cols) - set(train.columns)
if missing:
    raise ValueError(f"Features missing from train: {sorted(missing)}")

X = train.loc[:, feature_cols].copy()
y = train[targets[0]].copy()
pipe = Pipeline(steps)
pipe = mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(pipe, "model.pkl")
print(f"train rows={len(X)} features={len(feature_cols)}")
print("steps=" + ",".join(name for name, _ in steps))
