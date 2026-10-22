import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol)}")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]

train = pd.read_parquet(inputs["train"])
with open(inputs["06_preprocess_fit_preprocessing_Q5T3SXM3/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["07_features_select_features_Y03WXMRF/selector.pkl"], "rb") as f:
    selector = pickle.load(f)
json.loads(inputs["06_preprocess_fit_preprocessing_Q5T3SXM3/manifest.json"].read_text())
json.loads(inputs["07_features_select_features_Y03WXMRF/manifest.json"].read_text())

def names(obj):
    if obj is None:
        return None
    n = getattr(obj, "feature_names_in_", None)
    return list(n) if n is not None else None

def key_like(s):
    if len(s) == 0:
        return False
    if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
        return s.nunique(dropna=True) == len(s) and len(s) > 20
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.nunique(dropna=True) == len(s) and len(s) > 20
    return False

pre_names = names(pre)
sel_names = names(selector)
raw_cols = pre_names or sel_names
if raw_cols is None:
    blocked = set(targets + excluded + ([group_col] if group_col else []))
    raw_cols = [c for c in train.columns if c not in blocked and not key_like(train[c])]

if target not in train.columns:
    raise KeyError(f"Target {target!r} not in training columns")
missing = [c for c in raw_cols if c not in train.columns]
if missing:
    raise KeyError(f"Missing feature columns: {missing}")

if selector is not None and pre_names is not None and sel_names is not None and set(sel_names) == set(pre_names):
    frontend = selector
    raw_cols = sel_names
else:
    steps = []
    if pre is not None:
        steps.append(("preprocessor", pre))
    if selector is not None:
        steps.append(("selector", selector))
    frontend = Pipeline(steps) if len(steps) > 1 else steps[0][1]
    raw_cols = pre_names or raw_cols

model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
pipe = Pipeline([("frontend", frontend), ("model", model)])

X = train.loc[:, raw_cols]
y = train[target]
fitted = mlca_runtime.fit_record(pipe, X, partition="train", y=y)

mlca_runtime.save_model(fitted, "model.pkl")
print(f"train rows={len(X)} features={len(raw_cols)} target={target}")
print("saved model.pkl")
