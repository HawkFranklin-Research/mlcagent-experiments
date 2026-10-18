import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}. Available keys: {sorted(protocol)}")

task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task key 'primary_metric'. Available keys: {sorted(task)}")

targets = protocol["targets"]
if len(targets) != 1:
    raise ValueError(f"Expected exactly one target, got {targets}")
target = targets[0]
excluded = set(protocol["excluded_columns"])
group = protocol["group_column"]

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

def key_like(s):
    if s.nunique(dropna=False) != len(s):
        return False
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
        or pd.api.types.is_integer_dtype(s)
    )

computed_features = [
    c for c in train.columns
    if c != target and c not in excluded and c != group and not key_like(train[c])
]
if not computed_features:
    raise ValueError("No feature columns remain after exclusions")

with open(paths["05_preprocess_fit_preprocessing_5QSFC5J4/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(paths["06_features_select_features_3FVCT0MK/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_columns(obj):
    first = obj.steps[0][1] if isinstance(obj, Pipeline) and obj.steps else obj
    names = getattr(first, "feature_names_in_", None)
    if names is None:
        names = getattr(obj, "feature_names_in_", None)
    return None if names is None else [str(c) for c in names]

sel_in = input_columns(sel)
if isinstance(sel, Pipeline) and len(sel.steps) >= 2 and (sel_in is None or set(sel_in) == set(computed_features)):
    frontend = sel
elif sel_in is not None and set(sel_in) == set(computed_features):
    frontend = Pipeline([("selector", sel)])
else:
    frontend = Pipeline([("preprocessor", pre), ("selector", sel)])

first_step = frontend.steps[0][1] if isinstance(frontend, Pipeline) and frontend.steps else frontend
expected = getattr(first_step, "feature_names_in_", None)
expected = [str(c) for c in expected] if expected is not None else None

if expected is not None:
    missing = [c for c in expected if c not in train.columns]
    if missing:
        raise KeyError(f"Saved frontend expects missing columns: {missing}")
    bad = [c for c in expected if c == target or c in excluded or c == group]
    if bad:
        raise ValueError(f"Saved frontend expects excluded/target columns: {bad}")
    if set(expected) != set(computed_features):
        raise ValueError(
            f"Saved frontend expects {len(expected)} columns that differ from the computed feature schema"
        )
    feature_cols = expected
else:
    feature_cols = computed_features

X_train = train[feature_cols].copy()
y_train = train[target].copy()

def set_pandas(obj):
    if hasattr(obj, "set_output"):
        try:
            obj.set_output(transform="pandas")
        except Exception:
            pass

set_pandas(pre)
set_pandas(sel)

model = Pipeline(
    frontend.steps
    + [("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))]
)
mlca_runtime.fit_record(model, X_train, "train", y_train)
mlca_runtime.save_model(model, "model.pkl")

if task["primary_metric"] == "roc_auc" and len(model.classes_) == 2:
    pos = task.get("positive_label")
    if pos is None:
        pos = model.classes_[1]
    if pos not in list(model.classes_):
        raise ValueError(f"Positive label {pos!r} not in model classes {list(model.classes_)}")
    proba = model.predict_proba(val[feature_cols].copy())
    col = list(model.classes_).index(pos)
    auc = roc_auc_score((val[target] == pos).astype(int), proba[:, col])
    mlca_runtime.record_metric("roc_auc", float(auc), "val")
    print("val roc_auc", round(float(auc), 4))

print("train rows", len(train), "val rows", len(val), "features", len(feature_cols))
print("saved model.pkl")
