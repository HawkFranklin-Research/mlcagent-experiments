import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing required protocol key '{k}'. Available top-level keys: {sorted(d.keys())}")
    return d[k]

targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
group_col = req(protocol, "group_column")
task = req(protocol, "task")

if isinstance(targets, str):
    targets = [targets]
if isinstance(excluded, str):
    excluded = [excluded]
if excluded is None:
    excluded = []
if isinstance(group_col, str):
    group_cols = [group_col]
elif group_col is None:
    group_cols = []
else:
    group_cols = list(group_col)

train = pd.read_parquet(inputs["train"])
drop = set(targets) | set(excluded) | set(group_cols)
candidate = [c for c in train.columns if c not in drop]

def is_key_like(s, n):
    if s.nunique(dropna=False) != n:
        return False
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s.dtype):
        return True
    return False

feature_cols = [c for c in candidate if not is_key_like(train[c], len(train))]
if not feature_cols:
    raise ValueError("No feature columns remain after exclusions and key-like filtering.")
if len(targets) != 1:
    raise ValueError(f"Expected one target for binary task, got {targets}")
target_col = targets[0]
if target_col not in train.columns:
    raise KeyError(f"Target column {target_col} not in training table.")

with open(inputs["05_preprocess_fit_preprocessing_QB1BRK7H/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, "feature_names_in_"):
    saved_cols = list(preprocessor.feature_names_in_)
    if set(saved_cols) != set(feature_cols):
        missing = set(saved_cols) - set(feature_cols)
        extra = set(feature_cols) - set(saved_cols)
        raise ValueError(f"Saved preprocessor column mismatch. Missing: {missing}, extra: {extra}")
    feature_cols = saved_cols

X_train = train[feature_cols].copy()
y_train = train[target_col]

selector = SelectKBest(score_func=mutual_info_classif, k="all")
if isinstance(preprocessor, Pipeline):
    existing = set(preprocessor.named_steps)
    new_name = "select"
    while new_name in existing:
        new_name += "_"
    steps = list(preprocessor.steps) + [(new_name, selector)]
    pipeline = Pipeline(steps)
else:
    new_name = "select"
    pipeline = Pipeline([("preprocessor", preprocessor), (new_name, selector)])

fitted = mlca_runtime.fit_record(pipeline, X_train, "train", y_train)
sel = fitted.named_steps[new_name]
sel.k = min(20, len(sel.scores_))
mlca_runtime.save_model(fitted, "selector.pkl")
print(f"train rows={len(X_train)} features={len(feature_cols)} transformed={len(sel.scores_)} k={sel.k}")
