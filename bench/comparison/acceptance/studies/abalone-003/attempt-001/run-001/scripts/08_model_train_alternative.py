import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def req(key):
    if key not in protocol:
        raise ValueError(f"Missing '{key}'. Available top-level keys: {list(protocol.keys())}")
    return protocol[key]

targets = req("targets")
if isinstance(targets, str):
    targets = [targets]
excluded = req("excluded_columns") or []
group = req("group_column")
task = req("task")
train = pd.read_parquet(inputs["train"])

def is_key_like(s):
    if pd.api.types.is_string_dtype(s) or s.dtype.kind == "O":
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)
declared_features = [c for c in train.columns if c not in drop and not is_key_like(train[c])]

with open(inputs["05_preprocess_fit_preprocessing_QJRR3HF7/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_8C5RPJ11/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def contains_preprocessing(obj):
    if isinstance(obj, ColumnTransformer):
        return True
    if isinstance(obj, Pipeline):
        return any(contains_preprocessing(step) for _, step in obj.named_steps.items())
    return False

def first_feature_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            cols = first_feature_names(step)
            if cols is not None:
                return cols
    return None

model = RandomForestRegressor(n_estimators=100, random_state=42)
if sel is not None and contains_preprocessing(sel):
    steps = [("frontend", sel), ("model", model)]
    input_cols = first_feature_names(sel)
else:
    steps = []
    if pre is not None:
        steps.append(("preprocessor", pre))
    if sel is not None:
        steps.append(("selector", sel))
    steps.append(("model", model))
    input_cols = first_feature_names(pre) if pre is not None else None

pipeline = Pipeline(steps)
if input_cols is None:
    feature_columns = declared_features
else:
    bad = [c for c in input_cols if c in drop]
    if bad:
        raise ValueError(f"Frontend expects excluded/target columns: {bad}")
    missing = [c for c in input_cols if c not in train.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")
    feature_columns = list(input_cols)

X_train = train[feature_columns]
y_train = train[targets].apply(pd.to_numeric, errors="raise")
fitted = mlca_runtime.fit_record(pipeline, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "alternative_model.pkl")
print(f"train rows={len(train)}, features={len(feature_columns)}, targets={targets}")
