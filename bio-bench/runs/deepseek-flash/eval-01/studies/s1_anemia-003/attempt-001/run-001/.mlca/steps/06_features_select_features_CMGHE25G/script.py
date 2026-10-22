import mlca_runtime
import json
import pickle
import re

import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing {key!r}. Available top-level keys: {sorted(d.keys())}")
    return d[key]

targets = list(need(protocol, "targets"))
excluded = list(need(protocol, "excluded_columns"))
task = need(protocol, "task")
group = protocol.get("group_column", None)

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_2E873D38/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

exclude = set(targets) | set(excluded)
if group:
    exclude.add(group)

def is_keylike(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=True) == len(s) and len(s) > 0
    if pd.api.types.is_integer_dtype(s):
        name = str(s.name).lower()
        return s.nunique(dropna=True) == len(s) and bool(re.search(r"(^|_)(id|key|code|index)$", name))
    return False

candidate = [c for c in train.columns if c not in exclude and not is_keylike(train[c])]
feature_cols = candidate

if hasattr(pre, "feature_names_in_"):
    pre_cols = list(pre.feature_names_in_)
    missing = [c for c in pre_cols if c not in feature_cols]
    if missing:
        raise ValueError(f"Preprocessor expects columns outside the safe feature list: {missing}")
    feature_cols = pre_cols

bad = [c for c in feature_cols if c in exclude]
if bad:
    raise ValueError(f"Target/excluded/group columns reached features: {bad}")

X_train = train[feature_cols].copy()
y_train = train[targets[0]]

probe = mlca_runtime.fit_record(clone(pre), X_train, "train", y_train)
Xt = probe.transform(X_train)
n_features = int(Xt.shape[1])
k = min(20, n_features)
if k < 1:
    raise ValueError("No preprocessed features available for SelectKBest")

selector_pipe = Pipeline([
    ("preprocessor", clone(pre)),
    ("selectkbest", SelectKBest(score_func=mutual_info_classif, k=k)),
])
fitted = mlca_runtime.fit_record(selector_pipe, X_train, "train", y_train)

mlca_runtime.record_metric("n_features_raw", len(feature_cols), "train")
mlca_runtime.record_metric("n_features_preprocessed", n_features, "train")
mlca_runtime.record_metric("n_features_selected", k, "train")

mlca_runtime.save_model(fitted, "selector.pkl")
print(f"train rows={len(train)}, raw features={len(feature_cols)}")
print(f"preprocessed width={n_features}, selected k={k}")
print("saved selector.pkl")
