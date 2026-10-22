import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {sorted(d.keys())}")
    return d[k]

targets = need(protocol, "targets")
if isinstance(targets, str):
    targets = [targets]
excluded = need(protocol, "excluded_columns")
group = protocol.get("group_column")
target_col = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

with open(inputs["05_preprocess_fit_preprocessing_FMADE2YX/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_retain_features_SEM4Z3JD/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)

def input_columns(obj):
    if obj is None:
        return None
    cols = getattr(obj, "feature_names_in_", None)
    if cols is None and isinstance(obj, Pipeline) and obj.steps:
        cols = getattr(obj.steps[0][1], "feature_names_in_", None)
    return list(cols) if cols is not None else None

def contains_step(obj, target):
    if not isinstance(obj, Pipeline):
        return False
    for _, step in obj.steps:
        if step is target:
            return True
        if isinstance(step, Pipeline) and contains_step(step, target):
            return True
    return False

def wraps_preprocessing(obj):
    if not isinstance(obj, Pipeline):
        return False
    if contains_step(obj, preprocessor):
        return True
    for name, step in obj.steps:
        if name.lower().startswith("preprocess") or step.__class__.__name__ == "ColumnTransformer":
            return True
    return False

def is_key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return len(s) > 0 and s.nunique(dropna=False) == len(s)
    if s.dtype.kind in "iu":
        name = str(s.name).lower()
        return ("id" in name or "code" in name or "key" in name) and len(s) > 0 and s.nunique(dropna=False) == len(s)
    return False

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
candidate = [c for c in train.columns if c not in drop]
feature_cols = [c for c in candidate if not is_key_like(train[c])]

pre_cols = input_columns(preprocessor)
sel_cols = input_columns(selector)

model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

if wraps_preprocessing(selector):
    raw_cols = sel_cols if sel_cols is not None else (pre_cols if pre_cols is not None else feature_cols)
    pipeline = Pipeline([("selector_all", selector), ("model", model)])
else:
    raw_cols = pre_cols if pre_cols is not None else feature_cols
    pipeline = Pipeline([("preprocessor", preprocessor), ("selector", selector), ("model", model)])

missing = [c for c in raw_cols if c not in train.columns]
if missing:
    raise KeyError(f"Saved pipeline expects missing columns: {missing}")
leak = set(raw_cols) & drop
if leak:
    raise ValueError(f"Target or excluded columns reached features: {sorted(leak)}")

X = train[raw_cols]
y = train[target_col]
if target_col not in train.columns:
    raise KeyError(f"Missing target column: {target_col}")

mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(pipeline, "model.pkl")

print("train rows:", train.shape[0], "features:", len(raw_cols))
print("target:", target_col, "pipeline steps:", [name for name, _ in pipeline.steps])
print("saved model.pkl")
