import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
train = pd.read_parquet(inputs["train"])

with open(inputs["05_preprocess_fit_preprocessing_CSS2QFXR/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_2A3B8CCD/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def key_like(s):
    n = len(s)
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == n
    if pd.api.types.is_integer_dtype(s):
        name = str(s.name).lower()
        return any(k in name for k in ("id", "index", "key", "code")) and s.nunique(dropna=False) == n
    return False

base_drop = set(targets) | set(excluded)
if group:
    base_drop.add(group)
feature_cols = [c for c in train.columns if c not in base_drop]
feature_cols = [c for c in feature_cols if not key_like(train[c])]

def set_pandas(obj):
    if obj is not None and hasattr(obj, "set_output"):
        try:
            obj.set_output(transform="pandas")
        except Exception:
            pass

def is_raw_frontend(obj):
    fn = getattr(obj, "feature_names_in_", None)
    if fn is None:
        return False
    cols = set(map(str, fn))
    forbidden = set(targets) | set(excluded) | ({group} if group else set())
    return cols.issubset(set(train.columns)) and not (cols & forbidden)

def has_prep(obj):
    if not isinstance(obj, Pipeline):
        return False
    for step in obj.named_steps.values():
        if hasattr(step, "transformers") or has_prep(step):
            return True
    return False

use_sel_alone = sel is not None and (is_raw_frontend(sel) or has_prep(sel))
frontend = sel if use_sel_alone else pre
set_pandas(frontend)

fn = getattr(frontend, "feature_names_in_", None)
if fn is not None:
    feature_cols = list(map(str, fn))
    forbidden = set(targets) | set(excluded) | ({group} if group else set())
    if set(feature_cols) & forbidden:
        raise ValueError("Saved frontend expects target or excluded columns")
    missing = [c for c in feature_cols if c not in train.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

X = train[feature_cols].copy()
y = train[targets].copy()

clf = MultiOutputClassifier(LogisticRegression(max_iter=2000))
if use_sel_alone:
    pipe = Pipeline([("frontend", sel), ("clf", clf)])
else:
    steps = [("pre", pre)]
    if sel is not None:
        steps.append(("sel", sel))
    steps.append(("clf", clf))
    pipe = Pipeline(steps)

pipe = mlca_runtime.fit_record(pipe, X, partition="train", y=y)
mlca_runtime.save_model(pipe, "model.pkl")
print("fitted", "X", X.shape, "y", y.shape, "-> model.pkl")
