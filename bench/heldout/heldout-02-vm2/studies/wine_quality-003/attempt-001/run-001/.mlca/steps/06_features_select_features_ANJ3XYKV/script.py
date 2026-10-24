import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing {key}; available top-level keys: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target_col = targets[0]
drop_cols = set(targets) | set(excluded) | ({group} if group else set())

train = pd.read_parquet(inputs["train"])
if target_col not in train.columns:
    raise KeyError(f"Target {target_col!r} not in train")

def key_like(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
        return s.notna().all() and s.nunique() == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.notna().all() and s.nunique() == len(s)
    return False

feature_cols = [
    c for c in train.columns
    if c not in drop_cols and not key_like(train[c])
]

with open(inputs["05_preprocess_fit_preprocessing_HXN3H5DR/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    feature_cols = list(pre.feature_names_in_)
    if set(feature_cols) & drop_cols:
        raise ValueError("Saved preprocessor expects a target, excluded or group column")

X = train[feature_cols].copy()
y = train[target_col].copy()

try:
    n_transformed = pre.transform(X).shape[1]
except Exception:
    pre = mlca_runtime.fit_record(pre, X, "train", y)
    n_transformed = pre.transform(X).shape[1]

k = min(20, n_transformed)
selector = SelectKBest(score_func=mutual_info_classif, k=k)

if isinstance(pre, Pipeline):
    steps = [(name, est) for name, est in pre.steps if not isinstance(est, SelectKBest)]
    steps.append(("selectkbest", selector))
    model = Pipeline(steps)
else:
    model = Pipeline([("preprocessor", pre), ("selectkbest", selector)])

mlca_runtime.fit_record(model, X, "train", y)
mlca_runtime.save_model(model, "selector.pkl")

print(f"train rows: {len(X)}, raw features: {len(feature_cols)}")
print(f"transformed features: {n_transformed}, k: {k}")
print(f"selected features: {int(model.named_steps['selectkbest'].get_support().sum())}")
