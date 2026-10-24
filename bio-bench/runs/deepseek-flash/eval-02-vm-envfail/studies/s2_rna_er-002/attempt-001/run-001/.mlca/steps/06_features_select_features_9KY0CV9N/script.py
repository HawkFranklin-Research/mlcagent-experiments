import json
import pickle

import pandas as pd
import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise ValueError(f"Missing protocol keys {missing}; available keys: {sorted(protocol)}")

targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group_col = protocol["group_column"]
target = targets[0]

train = pd.read_parquet(inputs["train"])
exclude = set(targets) | excluded
if group_col:
    exclude.add(group_col)

with open(inputs["05_preprocess_fit_preprocessing_03EVNKD3/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    feature_cols = list(pre.feature_names_in_)
else:
    feature_cols = []
    n_rows = len(train)
    for c in train.columns:
        if c in exclude:
            continue
        s = train[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
            if s.nunique(dropna=True) == n_rows:
                continue
        elif pd.api.types.is_integer_dtype(s.dtype):
            if s.nunique(dropna=True) == n_rows:
                continue
        feature_cols.append(c)

bad = [c for c in feature_cols if c in exclude]
if bad:
    raise ValueError(f"Preprocessor expects excluded columns: {bad}")
missing_cols = [c for c in feature_cols if c not in train.columns]
if missing_cols:
    raise ValueError(f"Missing feature columns in train: {missing_cols}")

X_train = train[feature_cols].copy()
y_train = train[target].copy()

n_encoded = pre.transform(X_train).shape[1]
k = min(20, n_encoded)

pipe = Pipeline([
    ("preprocess", pre),
    ("select", SelectKBest(score_func=mutual_info_classif, k=k)),
])
mlca_runtime.fit_record(pipe, X_train, "train", y_train)

selected = int(pipe.named_steps["select"].get_support().sum())
mlca_runtime.record_metric("n_raw_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_encoded_features", float(n_encoded), "train")
mlca_runtime.record_metric("n_selected_features", float(selected), "train")
mlca_runtime.save_model(pipe, "selector.pkl")
print(f"raw={len(feature_cols)} encoded={n_encoded} selected={selected}")
