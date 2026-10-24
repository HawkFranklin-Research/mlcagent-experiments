import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
with open(inputs["_protocol"], "r") as f:
    proto = json.load(f)

targets = list(proto["targets"])
excluded = set(proto["excluded_columns"])
group = proto.get("group_column")
excluded_set = set(targets) | excluded | ({group} if group else set())

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_GK4CB9FZ/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_5542Z2W2/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def first_feature_names(obj):
    names = getattr(obj, "feature_names_in_", None)
    if names is not None:
        return list(names)
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            names = first_feature_names(step)
            if names is not None:
                return names
    return None

def contains_preprocessing(obj):
    if not isinstance(obj, Pipeline):
        return False
    for name, step in obj.named_steps.items():
        if name.lower() in {"preprocessor", "preprocess", "preprocessing"}:
            return True
        if step.__class__.__name__ == "ColumnTransformer":
            return True
    return False

sel_has_pre = contains_preprocessing(sel)
frontend = sel if sel_has_pre else pre
raw_cols = first_feature_names(frontend)
if raw_cols is None:
    raw_cols = [
        c for c in train.columns
        if c not in excluded_set
        and not (
            (train[c].dtype == object or pd.api.types.is_string_dtype(train[c]))
            and train[c].nunique() == len(train)
            and train[c].nunique() > 10
        )
    ]

bad = set(raw_cols) & excluded_set
missing = set(raw_cols) - set(train.columns)
if bad:
    raise ValueError(f"Feature columns overlap target/excluded/group: {sorted(bad)}")
if missing:
    raise ValueError(f"Feature columns missing from train: {sorted(missing)}")
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")

X = train[raw_cols].copy()
y = train[targets[0]].copy()
lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

if sel_has_pre:
    model = Pipeline(list(frontend.steps) + [("model", lr)])
else:
    steps = [("preprocess", pre)]
    if sel is not None:
        steps.append(("select", sel))
    steps.append(("model", lr))
    model = Pipeline(steps)

mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(model, "model.pkl")
print(f"train rows={len(train)}, features={len(raw_cols)}, target={targets[0]}")
print(f"pipeline steps={list(model.named_steps)}")
