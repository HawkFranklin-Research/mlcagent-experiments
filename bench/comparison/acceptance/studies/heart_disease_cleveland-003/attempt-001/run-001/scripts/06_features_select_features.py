import json
import pickle

import mlca_runtime
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
protocol = json.loads(inp["_protocol"].read_text())

needed = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in needed if k not in protocol]
if missing:
    raise KeyError(
        f"Missing protocol keys {missing}; available top-level keys: {list(protocol.keys())}"
    )

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol["group_column"]
_ = protocol["task"]

if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]

with open(inp["05_preprocess_fit_preprocessing_S0BP8ZJM/preprocessor.pkl"], "rb") as f:
    preproc = pickle.load(f)

drop = set(targets) | set(excluded)
if group_col is not None:
    drop.add(group_col)

n_rows = len(train)
feature_cols = []
for col in train.columns:
    if col in drop:
        continue
    s = train[col]
    if n_rows > 1 and s.nunique(dropna=True) == n_rows:
        lname = str(col).lower()
        is_obj = (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
        )
        if is_obj or ("id" in lname or "key" in lname or "code" in lname):
            continue
    feature_cols.append(col)

if not feature_cols:
    raise ValueError("No feature columns remain after exclusions")

input_obj = preproc
if isinstance(preproc, Pipeline) and preproc.steps and isinstance(preproc.steps[-1][1], SelectKBest):
    frontend = preproc[:-1]
else:
    frontend = preproc

expected = None
if hasattr(input_obj, "feature_names_in_"):
    expected = list(input_obj.feature_names_in_)
elif hasattr(frontend, "feature_names_in_"):
    expected = list(frontend.feature_names_in_)

if expected is not None:
    bad = [c for c in expected if c in drop or c not in feature_cols]
    if bad:
        raise ValueError(f"Preprocessor expects columns outside the allowed feature set: {bad}")
    feature_cols = expected

X_train = train[feature_cols].copy()
y_train = train[target].copy()


def n_output_features(obj, X):
    try:
        return len(obj.get_feature_names_out())
    except Exception:
        try:
            return obj.transform(X.iloc[:1]).shape[1]
        except Exception:
            return len(X.columns)


n_features = n_output_features(frontend, X_train)
k = min(20, int(n_features))
if k < 1:
    raise ValueError(f"Invalid number of features to select: {k}")

selector = SelectKBest(score_func=mutual_info_classif, k=k)
if isinstance(frontend, Pipeline):
    pipeline = Pipeline(list(frontend.steps) + [("select_k_best", selector)])
else:
    pipeline = Pipeline([("preprocessor", frontend), ("select_k_best", selector)])

pipeline = mlca_runtime.fit_record(pipeline, X_train, "train", y_train)
mlca_runtime.save_model(pipeline, "selector.pkl")

selected = int(pipeline.named_steps["select_k_best"].get_support().sum())
print(f"train rows: {len(train)}")
print(f"feature cols: {len(feature_cols)}")
print(f"preprocessed width: {n_features}, k: {k}")
print(f"selected: {selected}")
