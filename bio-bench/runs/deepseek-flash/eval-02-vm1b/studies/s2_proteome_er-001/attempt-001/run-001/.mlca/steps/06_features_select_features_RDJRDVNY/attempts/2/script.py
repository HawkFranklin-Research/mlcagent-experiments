import mlca_runtime
import json
import pickle
from functools import partial

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
train = pd.read_parquet(paths["train"])

targets = list(protocol["targets"])
target_col = targets[0]
y = train[target_col]
excluded = set(protocol.get("excluded_columns") or [])
group_col = protocol.get("group_column")

with open(paths["05_preprocess_fit_preprocessing_5K21TQY3/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    missing = [c for c in pre.feature_names_in_ if c not in train.columns]
    if missing:
        raise KeyError(f"missing preprocessor features: {missing}")
    feature_cols = list(pre.feature_names_in_)
else:
    feature_cols = [c for c in train.columns if c not in set(targets) | excluded and c != group_col]
    n = len(train)
    kept = []
    for c in feature_cols:
        s = train[c]
        if s.nunique(dropna=True) == n and "id" in c.lower() and (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_integer_dtype(s)
            or pd.api.types.is_string_dtype(s)
        ):
            continue
        kept.append(c)
    feature_cols = kept

X = train[feature_cols]

try:
    encoded_names = pre.get_feature_names_out()
except Exception:
    pre = mlca_runtime.fit_record(pre, X, "train", y=y)
    encoded_names = pre.get_feature_names_out()

n_features = len(encoded_names)
k = min(20, n_features)
score_func = partial(mutual_info_classif, random_state=42)
pipe = Pipeline([
    ("preprocessor", pre),
    ("selector", SelectKBest(score_func=score_func, k=k)),
])
fitted = mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(fitted, "selector.pkl")
print(f"train_rows={len(train)} raw_features={len(feature_cols)} encoded_features={n_features} k={k}")
