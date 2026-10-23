import mlca_runtime
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(Path(inputs["_protocol"]).read_text())


def need(mapping, key):
    if key not in mapping:
        raise KeyError(f"Missing {key!r}; available top-level keys: {sorted(mapping.keys())}")
    return mapping[key]


targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group_col = need(protocol, "group_column")
task = need(protocol, "task")
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]

train = pd.read_parquet(inputs["train"])
pre = pickle.loads(Path(inputs["06_preprocess_fit_preprocessing_PDW4RHJ4/preprocessor.pkl"]).read_bytes())
_manifest = json.loads(Path(inputs["06_preprocess_fit_preprocessing_PDW4RHJ4/manifest.json"]).read_text())

exclude = set(targets) | set(excluded)
if group_col is not None:
    exclude.add(group_col)


def is_keylike(series):
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return series.nunique(dropna=False) == len(series)
    if pd.api.types.is_integer_dtype(series):
        return series.nunique(dropna=False) == len(series)
    return False


allowed = [c for c in train.columns if c not in exclude and not is_keylike(train[c])]
if hasattr(pre, "feature_names_in_"):
    expected = list(pre.feature_names_in_)
    bad = [c for c in expected if c in exclude]
    if bad:
        raise ValueError(f"Preprocessor expects excluded columns: {bad}")
    missing = [c for c in expected if c not in train.columns]
    if missing:
        raise ValueError(f"Preprocessor expects missing columns: {missing}")
    feature_columns = expected
else:
    feature_columns = allowed

X = train[feature_columns]
y = train[target]

try:
    n_features = pre.transform(X.head(1)).shape[1]
except Exception:
    n_features = len(feature_columns)

k = min(20, n_features)
selector = SelectKBest(score_func=mutual_info_classif, k=k)

if isinstance(pre, Pipeline):
    steps = list(pre.steps)
    name = "selectkbest"
    names = {n for n, _ in steps}
    while name in names:
        name += "_"
    model = Pipeline(steps + [(name, selector)])
else:
    model = Pipeline([("preprocess", pre), ("selectkbest", selector)])

mlca_runtime.fit_record(model, X, partition="train", y=y)
mlca_runtime.record_metric("n_candidate_features", float(n_features), split="train")
mlca_runtime.record_metric("n_selected_features", float(k), split="train")
mlca_runtime.save_model(model, "selector.pkl")
print(f"train_rows={len(train)} candidate_features={n_features} k={k}")
print("saved selector.pkl")
