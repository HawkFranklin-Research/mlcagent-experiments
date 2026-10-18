import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {sorted(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol["group_column"]
task = protocol["task"]
if not isinstance(task, dict) or "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; task keys: {sorted(task.keys()) if isinstance(task, dict) else type(task)}")

train = pd.read_parquet(inputs["train"])

def is_key_like(s: pd.Series) -> bool:
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False

drop_cols = set(targets) | set(excluded)
if group_col is not None:
    drop_cols.add(group_col)
candidate_cols = [c for c in train.columns if c not in drop_cols]
key_like = [c for c in candidate_cols if is_key_like(train[c])]
feature_cols = [c for c in candidate_cols if c not in set(key_like)]

with open(inputs["05_preprocess_fit_preprocessing_0CQ66EDA/preprocessor.pkl"], "rb") as f:
    preproc = pickle.load(f)

input_cols = feature_cols
if hasattr(preproc, "feature_names_in_"):
    input_cols = list(preproc.feature_names_in_)
    bad = set(input_cols) & drop_cols
    if bad:
        raise ValueError(f"Preprocessor input includes target/excluded/group columns: {sorted(bad)}")

missing_cols = [c for c in input_cols if c not in train.columns]
if missing_cols:
    raise KeyError(f"Missing input columns in train: {missing_cols}")

X = train.loc[:, input_cols]
passthrough = FunctionTransformer(func=None, validate=False)

if isinstance(preproc, Pipeline):
    selector_all = Pipeline(list(preproc.steps) + [("selector_all", passthrough)])
else:
    selector_all = Pipeline([("preprocessor", preproc), ("selector_all", passthrough)])

mlca_runtime.fit_record(selector_all, X, "train")
mlca_runtime.save_model(selector_all, "selector_all.pkl")

print(f"train rows: {len(train)}")
print(f"features: {len(input_cols)}")
print("saved selector_all.pkl")
