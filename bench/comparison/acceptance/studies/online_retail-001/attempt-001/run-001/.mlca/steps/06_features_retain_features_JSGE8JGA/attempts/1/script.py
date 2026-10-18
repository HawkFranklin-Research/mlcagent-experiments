import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()

def load_json(path):
    return json.loads(path.read_text())

protocol = load_json(paths["_protocol"])
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(
        f"Missing protocol keys {missing}; available top-level keys: {sorted(protocol.keys())}"
    )

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]
task = protocol["task"]

train = pd.read_parquet(paths["train"])
manifest = load_json(paths["05_preprocess_fit_preprocessing_GGPQ6YHJ/manifest.json"])
with open(paths["05_preprocess_fit_preprocessing_GGPQ6YHJ/preprocessor.pkl"], "rb") as f:
    preproc = pickle.load(f)

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

def is_key_like(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_float_dtype(s):
        return False
    dtype_ok = (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
        or pd.api.types.is_integer_dtype(s)
    )
    return dtype_ok and s.nunique(dropna=False) == len(s)

feature_cols = [
    c for c in train.columns if c not in drop and not is_key_like(train[c])
]

def feature_names_in(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline) and obj.steps:
        return feature_names_in(obj.steps[0][1])
    return None

pre_in = feature_names_in(preproc)
if pre_in is not None:
    bad = [c for c in pre_in if c in drop]
    if bad:
        raise ValueError(f"Saved preprocessor expects excluded/target/group columns: {bad}")
    absent = [c for c in pre_in if c not in train.columns]
    if absent:
        raise ValueError(f"Saved preprocessor expects columns absent from train: {absent}")
    input_cols = pre_in
else:
    input_cols = feature_cols

X = train.loc[:, input_cols]

if isinstance(preproc, Pipeline) and preproc.steps and preproc.steps[-1][1] == "passthrough":
    selector = preproc
else:
    selector = Pipeline([("preprocessor", preproc), ("selector", "passthrough")])

selector = mlca_runtime.fit_record(selector, X, partition="train")
mlca_runtime.save_model(selector, "selector_all.pkl")

print("train rows:", len(train))
print("feature columns:", len(input_cols))
print("saved selector_all.pkl")
