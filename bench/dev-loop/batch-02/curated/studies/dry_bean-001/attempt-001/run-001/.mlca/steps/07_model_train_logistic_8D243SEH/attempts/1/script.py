import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing {k}. Available top-level keys: {list(protocol.keys())}")
targets = protocol["targets"]
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]
task = protocol["task"]
if len(targets) != 1:
    raise ValueError(f"Expected exactly one target, got {targets}")
target = targets[0]

train = pd.read_parquet(inputs["train"])
if target not in train.columns:
    raise KeyError(f"Target {target} not in training table")

def is_keylike(s):
    n = len(s)
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == n
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == n
    return False

drop = set(targets) | set(excluded) | ({group} if group is not None else set())
feature_cols = [c for c in train.columns if c not in drop and not is_keylike(train[c])]
if not feature_cols:
    raise ValueError("No feature columns available")

with open(inputs["05_preprocess_fit_preprocessing_CQ5BZKTY/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_XJNGW7H2/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    expected = list(pre.feature_names_in_)
    missing = [c for c in expected if c not in train.columns]
    if missing:
        raise KeyError(f"Preprocessor expects missing columns: {missing}")
    if any(c in drop for c in expected):
        raise ValueError("Preprocessor expects a target/excluded/group column")
    feature_cols = expected

def accepts_raw(obj):
    if isinstance(obj, Pipeline):
        for _, step in obj.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                if list(step.feature_names_in_) == feature_cols:
                    return True
        return False
    return hasattr(obj, "feature_names_in_") and list(obj.feature_names_in_) == feature_cols

if isinstance(sel, Pipeline) and accepts_raw(sel):
    frontend = sel
elif accepts_raw(pre):
    if isinstance(pre, Pipeline) and any("Select" in type(s).__name__ for s in pre.named_steps.values()):
        frontend = pre
    else:
        frontend = Pipeline([("pre", pre), ("sel", sel)])
else:
    frontend = Pipeline([("pre", pre), ("sel", sel)])

model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
pipeline = Pipeline([("frontend", frontend), ("model", model)])

if hasattr(pipeline, "set_output"):
    try:
        pipeline.set_output(transform="pandas")
    except Exception:
        pass

X = train[feature_cols]
y = train[target]
pipeline = mlca_runtime.fit_record(pipeline, X, "train", y)
mlca_runtime.save_model(pipeline, "model.pkl")
acc = float(pipeline.score(X, y))
mlca_runtime.record_metric("train_accuracy", acc, "train")
print(f"features: {len(feature_cols)}")
print(f"train rows: {len(train)}")
print(f"train accuracy: {acc:.4f}")
