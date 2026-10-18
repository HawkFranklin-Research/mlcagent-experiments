import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

protocol = json.loads(inp["_protocol"].read_text())
for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
for key in [
    "05_preprocess_fit_preprocessing_Y0BA94GV/manifest.json",
    "06_features_retain_features_J9XJJ7CH/manifest.json",
]:
    json.loads(inp[key].read_text())

with open(inp["05_preprocess_fit_preprocessing_Y0BA94GV/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_retain_features_J9XJJ7CH/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=True) == len(s) and len(s) > 1
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == len(s) and len(s) > 1
    return False

def cols_of(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def is_transformer(est):
    return hasattr(est, "transform")

def last_is_transformer(pipe):
    return isinstance(pipe, Pipeline) and is_transformer(pipe.named_steps[list(pipe.named_steps)[-1]])

computed = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group and not key_like(train[c])
]
feature_cols = computed
for obj in [sel, pre]:
    cols = cols_of(obj)
    if cols is not None and set(cols).issubset(train.columns):
        bad = set(cols) & (set(targets) | set(excluded) | ({group} if group else set()))
        if not bad and not any(key_like(train[c]) for c in cols):
            feature_cols = list(cols)
            break

steps = []
frontend = None
if last_is_transformer(sel) and cols_of(sel) == feature_cols:
    frontend = sel
if frontend is None and is_transformer(pre) and cols_of(pre) == feature_cols:
    frontend = pre
    sel_cols = cols_of(sel)
    if (not isinstance(sel, Pipeline)) and is_transformer(sel) and sel_cols is not None:
        contains = isinstance(pre, Pipeline) and any(isinstance(st, type(sel)) for st in pre.named_steps.values())
        if not contains:
            steps.append(("selector", sel))
if frontend is not None:
    steps.insert(0, ("frontend", frontend))
steps.append(("model", LinearRegression()))
pipe = Pipeline(steps)

X_train = train[feature_cols]
y_train = train[targets]
fitted = mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "model.pkl")

print(f"train={train.shape} val={val.shape} features={len(feature_cols)}")
print(f"targets={targets}")
print(f"steps={list(fitted.named_steps)}")
