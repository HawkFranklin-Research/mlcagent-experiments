import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

if "targets" not in protocol:
    raise KeyError(f"Missing 'targets'; available top-level keys: {list(protocol)}")
if "task" not in protocol:
    raise KeyError(f"Missing 'task'; available top-level keys: {list(protocol)}")

targets = list(protocol["targets"])
if not targets:
    raise ValueError("No targets declared in _protocol['targets']")
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(inputs["train"])
if target not in train.columns:
    raise KeyError(f"Target {target!r} not found in train columns")

with open(inputs["05_preprocess_fit_preprocessing_PTKJZF9S/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_00QZZ2DE/selector.pkl"], "rb") as f:
    sel = pickle.load(f)


def input_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline) and obj.named_steps:
        first = next(iter(obj.named_steps.values()))
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
    return None


def output_names(obj):
    try:
        return list(obj.get_feature_names_out())
    except Exception:
        return None


def contains_equivalent(pipeline, other):
    if not isinstance(pipeline, Pipeline):
        return False
    for step in pipeline.named_steps.values():
        if step is other:
            return True
        a = getattr(step, "feature_names_in_", None)
        b = getattr(other, "feature_names_in_", None)
        if a is not None and b is not None and list(a) == list(b) and type(step) is type(other):
            return True
    return False


pre_in = input_names(pre)
sel_in = input_names(sel)
pre_out = output_names(pre)

if isinstance(sel, Pipeline) and len(sel.named_steps) > 1 and pre_in and sel_in and list(pre_in) == list(sel_in):
    frontend = sel
elif contains_equivalent(sel, pre):
    frontend = sel
elif sel_in is not None and pre_out is not None and set(sel_in).issubset(set(pre_out)):
    frontend = Pipeline([("preprocessor", pre), ("selector", sel)])
elif sel_in is None:
    frontend = Pipeline([("preprocessor", pre), ("selector", sel)])
elif pre_in and sel_in and list(pre_in) == list(sel_in):
    frontend = sel
else:
    frontend = Pipeline([("preprocessor", pre), ("selector", sel)])

front_in = input_names(frontend)
if front_in:
    bad = [c for c in front_in if c in targets or c in excluded or c == group]
    if bad:
        raise ValueError(f"Saved frontend expects target/excluded/group columns: {bad}")
    feature_cols = [c for c in front_in if c in train.columns]
else:
    feature_cols = []
    for c in train.columns:
        if c in targets or c in excluded or c == group:
            continue
        s = train[c]
        lname = c.lower()
        if any(t in lname for t in ["id", "key", "code", "record", "subject", "patient"]):
            if s.nunique(dropna=True) == len(s) and len(s) > 1:
                continue
        if (pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)):
            if s.nunique(dropna=True) == len(s) and len(s) > 1:
                continue
        feature_cols.append(c)

pipeline = Pipeline([
    ("frontend", frontend),
    ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
])

X = train[feature_cols]
y = train[target]
mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(pipeline, "model.pkl")
mlca_runtime.record_metric("train_rows", float(len(train)), "train")
mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
print(f"train rows={len(train)} features={len(feature_cols)}")
print("saved model.pkl")
