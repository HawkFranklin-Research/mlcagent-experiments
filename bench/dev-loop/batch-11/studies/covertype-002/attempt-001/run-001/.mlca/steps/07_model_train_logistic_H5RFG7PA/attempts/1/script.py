import json
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
for key in ["targets"]:
    if key not in proto:
        raise KeyError(f"Missing _protocol key {key!r}; available keys: {sorted(proto)}")

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


candidate_cols = []
for col in train.columns:
    if col in targets or col in excluded or (group is not None and col == group):
        continue
    if is_key_like(train[col]):
        continue
    candidate_cols.append(col)

with open(inp["05_preprocess_fit_preprocessing_8APC2384/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_select_features_23ZRTGDS/selector.pkl"], "rb") as f:
    sel = pickle.load(f)


def names_in(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            got = names_in(step)
            if got is not None:
                return got
    return None


def names_out(obj):
    if hasattr(obj, "get_feature_names_out"):
        try:
            return list(obj.get_feature_names_out())
        except Exception:
            return None
    return None


def check_no_bad(cols, label):
    banned = set(targets) | set(excluded)
    if group is not None:
        banned.add(group)
    bad = set(cols) & banned
    if bad:
        raise ValueError(f"{label} expects banned columns: {sorted(bad)}")


pre_in = names_in(pre)
sel_in = names_in(sel)
pre_out = names_out(pre)
clf = LogisticRegression(
    C=1.0,
    class_weight="balanced",
    max_iter=2000,
    solver="saga",
)

if sel_in is not None and pre_out is not None and list(sel_in) == list(pre_out):
    steps = [("preprocessing", pre), ("selector", sel), ("classifier", clf)]
elif sel_in is not None and set(sel_in) <= set(candidate_cols):
    if isinstance(sel, Pipeline) and pre_in is not None and list(sel_in) == list(pre_in):
        steps = [("selector_pipeline", sel), ("classifier", clf)]
    elif pre_in is not None and list(sel_in) == list(pre_in):
        steps = [("preprocessing", pre), ("classifier", clf)]
    else:
        steps = [("selector_pipeline", sel), ("classifier", clf)]
else:
    steps = [("preprocessing", pre), ("selector", sel), ("classifier", clf)]

front = steps[0][1]
front_in = names_in(front)
if front_in is not None:
    check_no_bad(front_in, steps[0][0])
    missing = set(front_in) - set(train.columns)
    if missing:
        raise ValueError(f"{steps[0][0]} needs missing columns: {sorted(missing)}")
    extra = set(front_in) - set(candidate_cols)
    if extra:
        raise ValueError(f"{steps[0][0]} needs non-feature columns: {sorted(extra)}")
    feature_cols = list(front_in)
else:
    feature_cols = list(candidate_cols)

width = len(pre_out) if pre_out is not None else None
if width is not None and width <= 300:
    for _, step in steps[:-1]:
        if hasattr(step, "set_output"):
            try:
                step.set_output(transform="pandas")
            except Exception:
                pass

pipe = Pipeline(steps)
X = train.loc[:, feature_cols].copy()
y = train[targets[0]].copy()
pipe = mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(pipe, "model.pkl")
print(f"train rows={len(X)} features={len(feature_cols)}")
print("steps=" + ",".join(name for name, _ in steps))
