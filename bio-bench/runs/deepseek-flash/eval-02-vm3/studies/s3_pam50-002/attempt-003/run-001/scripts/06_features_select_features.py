import json
import pickle

import mlca_runtime
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
required = {"targets", "excluded_columns", "task"}
missing = required - set(protocol)
if missing:
    raise KeyError(f"Missing required protocol keys {sorted(missing)}. Available top-level keys: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")
target_col = targets[0]

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_FS0JB8RV/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, "feature_names_in_"):
    feature_cols = [str(c) for c in preprocessor.feature_names_in_]
    missing_cols = [c for c in feature_cols if c not in train.columns]
    if missing_cols:
        raise KeyError(f"Preprocessor expects columns absent from train: {missing_cols[:10]}")
else:
    forbidden = set(targets) | set(excluded)
    if group_col:
        forbidden.add(group_col)
    feature_cols = []
    for c in train.columns:
        if c in forbidden:
            continue
        s = train[c]
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
            continue
        feature_cols.append(c)

bad = [c for c in feature_cols if c in targets or c in excluded or c == group_col]
if bad:
    raise ValueError(f"Feature list contains target/excluded/group columns: {bad[:10]}")

X = train[feature_cols]
y = train[target_col]

try:
    n_features = len(preprocessor.get_feature_names_out())
except Exception:
    n_features = preprocessor.transform(X.head(min(5, len(X)))).shape[1]

k = min(20, n_features)
if k < 1:
    raise ValueError("Preprocessing produced no features for SelectKBest")

selector = SelectKBest(score_func=mutual_info_classif, k=k)
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("select", selector),
])

fitted = mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(fitted, "selector.pkl")

print(f"train={train.shape}, raw_features={len(feature_cols)}, transformed_features={n_features}, k={k}")
