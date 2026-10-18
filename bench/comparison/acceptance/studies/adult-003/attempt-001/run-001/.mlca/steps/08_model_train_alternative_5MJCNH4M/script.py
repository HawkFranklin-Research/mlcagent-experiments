import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 42
mlca_runtime.set_seed(SEED)

inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
task = protocol["task"]
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]

train = pd.read_parquet(inp["train"])
y = train[target]

def feature_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

with open(inp["07_model_train_logistic_8BYRNT7K/model.pkl"], "rb") as f:
    saved = pickle.load(f)

base_steps = None
feature_cols = feature_names(saved)

if isinstance(saved, Pipeline):
    steps = list(saved.steps)
    if steps and (hasattr(steps[-1][1], "predict") or hasattr(steps[-1][1], "predict_proba")):
        steps = steps[:-1]
    base_steps = steps
    if feature_cols is None:
        for _, step in saved.steps:
            feature_cols = feature_names(step)
            if feature_cols is not None:
                break
else:
    with open(inp["05_preprocess_fit_preprocessing_599QM3CC/preprocessor.pkl"], "rb") as f:
        pre = pickle.load(f)
    with open(inp["06_features_select_features_Y69A7R1X/selector.pkl"], "rb") as f:
        sel = pickle.load(f)
    if isinstance(sel, Pipeline) and any(name in ("preprocessor", "preprocess") for name, _ in sel.steps):
        base_steps = [("selector", sel)]
    else:
        base_steps = [("preprocessor", pre), ("selector", sel)]
    feature_cols = feature_names(pre) or feature_names(sel) or feature_names(saved)

if feature_cols is None:
    group = protocol.get("group_column")
    feature_cols = []
    for col in train.columns:
        if col in targets or col in excluded or col == group:
            continue
        s = train[col]
        if s.is_unique and (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_integer_dtype(s)
        ):
            continue
        feature_cols.append(col)

missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise KeyError(f"Missing declared feature columns in train: {missing}")

X = train[feature_cols].copy()
model = HistGradientBoostingClassifier(random_state=SEED)
pipeline = Pipeline(list(base_steps) + [("classifier", model)])
pipeline = mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(pipeline, "alternative_model.pkl")

print(f"train rows={len(train)} features={len(feature_cols)}")
print(f"saved alternative_model.pkl steps={[name for name, _ in pipeline.steps]}")
