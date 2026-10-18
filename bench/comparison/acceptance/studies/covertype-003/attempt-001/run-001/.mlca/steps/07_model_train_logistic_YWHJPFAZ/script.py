import json
import pickle
import re

import mlca_runtime
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

required_keys = ["targets", "excluded_columns", "task"]
if any(k not in protocol for k in required_keys):
    raise KeyError(f"Protocol missing {required_keys}; available: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
_ = protocol["task"]["primary_metric"]
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
y_col = targets[0]

train = pd.read_parquet(inp["train"])
if y_col not in train.columns:
    raise KeyError(f"Target {y_col!r} not in train columns")

id_pat = re.compile(r"(^|_)(id|key|row|index|uuid|guid)(_|$)", re.I)

def key_like(s, name):
    if pd.api.types.is_integer_dtype(s):
        return bool(id_pat.search(name)) and s.nunique(dropna=False) == len(s)
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    return s.nunique(dropna=False) == len(s)

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    if not key_like(train[c], c):
        feature_cols.append(c)

X_train = train.loc[:, feature_cols]
y_train = train[y_col]

def load_pkl(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("05_preprocess_fit_preprocessing_A5VJ9RPH/preprocessor.pkl")
sel = load_pkl("06_features_select_features_KR1QEBV4/selector.pkl")

def contains(obj, target):
    if obj is target:
        return True
    if isinstance(obj, Pipeline):
        return any(step is target or contains(step, target) for _, step in obj.steps)
    return False

def input_names(obj):
    n = getattr(obj, "feature_names_in_", None)
    if n is None and isinstance(obj, Pipeline) and obj.steps:
        n = getattr(obj.steps[0][1], "feature_names_in_", None)
    return list(n) if n is not None else None

def looks_like_preprocessing(obj):
    if isinstance(obj, Pipeline):
        if not obj.steps:
            return False
        name, first = obj.steps[0]
        return "pre" in name.lower() or isinstance(first, (ColumnTransformer, Pipeline)) or hasattr(first, "transformers")
    return isinstance(obj, ColumnTransformer)

raw_set = set(feature_cols)
sn = input_names(sel)
selector_is_complete = (
    isinstance(sel, Pipeline)
    and len(sel.steps) >= 2
    and looks_like_preprocessing(sel)
    and (sn is None or (len(sn) == len(feature_cols) and set(sn) == raw_set))
)

if contains(sel, pre) or selector_is_complete:
    base = sel
elif contains(pre, sel):
    base = pre
else:
    base = Pipeline([("preprocessing", pre), ("selector", sel)])

if hasattr(base, "set_output"):
    try:
        base.set_output(transform="pandas")
    except Exception:
        pass

clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
if isinstance(base, Pipeline):
    pipe = Pipeline(list(base.steps) + [("classifier", clf)])
else:
    pipe = Pipeline([("preprocessing", base), ("classifier", clf)])

fitted = mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "model.pkl")

print(f"features={len(feature_cols)} rows={len(X_train)} target={y_col}")
print(f"steps={list(fitted.named_steps)}")
print("saved model.pkl")
