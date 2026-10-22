import json
import pickle
from functools import partial

import pandas as pd
import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_YDX4CF7Z/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    feature_cols = list(pre.feature_names_in_)
    missing = [c for c in feature_cols if c not in train.columns]
    if missing:
        raise KeyError(f"Preprocessor expects missing columns: {missing}")
else:
    feature_cols = []
    for c in train.columns:
        if c in targets or c in excluded or c == group:
            continue
        s = train[c]
        if s.is_unique and (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_integer_dtype(s)
        ):
            continue
        feature_cols.append(c)

bad = [c for c in feature_cols if c in targets or c in excluded or c == group]
if bad:
    raise ValueError(f"Feature list contains target/excluded/group columns: {bad}")

y = train[targets[0]]
X = train[feature_cols]

if hasattr(pre, "get_feature_names_out"):
    try:
        n_features = len(pre.get_feature_names_out(feature_cols))
    except TypeError:
        n_features = len(pre.get_feature_names_out())
    except Exception:
        n_features = pre.transform(X.iloc[:1]).shape[1]
else:
    n_features = pre.transform(X.iloc[:1]).shape[1]

k = min(20, n_features)
selector = SelectKBest(score_func=partial(mutual_info_classif, random_state=42), k=k)
pipe = Pipeline([("preprocessor", pre), ("selector", selector)])
pipe = mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(pipe, "selector.pkl")

print(f"features={len(feature_cols)} transformed={n_features} k={k}")
