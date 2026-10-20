import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    group_col = protocol["group_column"]
    task = protocol["task"]
except KeyError as exc:
    raise KeyError(f"Missing protocol key {exc}; available top-level keys: {list(protocol.keys())}") from exc

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_8XQPADGG/preprocessor.pkl"], "rb") as fh:
    preproc = pickle.load(fh)
if isinstance(preproc, dict) and "preprocessor" in preproc:
    preproc = preproc["preprocessor"]

drop_cols = set(targets) | set(excluded)
if group_col is not None:
    drop_cols.add(group_col)

def is_key_like(s: pd.Series) -> bool:
    n = len(s)
    if n == 0:
        return False
    name = str(s.name).lower()
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        if s.nunique(dropna=True) == n:
            return True
    if any(tok in name for tok in ("id", "uuid", "key", "code", "index")):
        if s.nunique(dropna=True) == n:
            return True
    return False

feature_cols = [c for c in train.columns if c not in drop_cols and not is_key_like(train[c])]
expected = getattr(preproc, "feature_names_in_", None)
if expected is not None:
    expected = list(expected)
    missing = [c for c in expected if c not in train.columns]
    bad = [c for c in expected if c in drop_cols]
    if missing or bad:
        raise ValueError(f"Saved preprocessor expects invalid columns: missing={missing}, bad={bad}")
    feature_cols = expected

X_train = train[feature_cols]
y_train = train[targets[0]]

try:
    n_transformed = len(preproc.get_feature_names_out())
except Exception:
    n_transformed = len(feature_cols)
k = max(1, min(20, n_transformed))

if isinstance(preproc, Pipeline) and any(isinstance(step, SelectKBest) for _, step in preproc.steps):
    steps = [
        (name, SelectKBest(mutual_info_classif, k=k) if isinstance(step, SelectKBest) else step)
        for name, step in preproc.steps
    ]
    pipe = Pipeline(steps)
else:
    pipe = Pipeline([
        ("preprocessor", preproc),
        ("selector", SelectKBest(mutual_info_classif, k=k)),
    ])

mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
mlca_runtime.save_model(pipe, "selector.pkl")

print(f"targets={targets}")
print(f"features={len(feature_cols)}")
print(f"transformed={n_transformed}")
print(f"k={k}")
print("saved selector.pkl")
