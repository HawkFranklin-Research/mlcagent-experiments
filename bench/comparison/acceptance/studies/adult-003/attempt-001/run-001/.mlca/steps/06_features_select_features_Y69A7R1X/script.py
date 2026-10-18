import json
import pickle
from functools import partial

import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
targets = protocol["targets"]
if isinstance(targets, str):
    targets = [targets]
excluded = protocol.get("excluded_columns", [])
if isinstance(excluded, str):
    excluded = [excluded]
group_col = protocol.get("group_column")

train = pd.read_parquet(paths["train"])
with open(paths["05_preprocess_fit_preprocessing_599QM3CC/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

def is_key_like(s):
    nm = str(s.name).lower()
    if any(tok in nm for tok in ("id", "key", "index", "code")):
        return True
    if s.notna().all() and s.nunique() == len(s):
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            return True
        if pd.api.types.is_integer_dtype(s):
            vals = np.sort(s.dropna().to_numpy())
            if len(vals) and vals[0] == 0 and np.array_equal(vals, np.arange(len(vals))):
                return True
    return False

drop = set(targets) | set(excluded)
if group_col is not None:
    drop.add(group_col)
feature_cols = [c for c in train.columns if c not in drop and not is_key_like(train[c])]

if hasattr(pre, "feature_names_in_"):
    pre_cols = list(pre.feature_names_in_)
    if set(pre_cols) != set(feature_cols):
        raise ValueError(
            f"Preprocessor input columns mismatch: "
            f"missing={sorted(set(feature_cols) - set(pre_cols))}, "
            f"extra={sorted(set(pre_cols) - set(feature_cols))}"
        )
    input_cols = pre_cols
else:
    input_cols = feature_cols

k = min(20, len(input_cols))
score = partial(mutual_info_classif, random_state=42)
selector = SelectKBest(score_func=score, k=k)
pipeline = Pipeline([("preprocessor", pre), ("selector", selector)])

X_train = train[input_cols].copy()
y_train = train[targets[0]]
mlca_runtime.fit_record(pipeline, X_train, "train", y_train)

n_selected = int(pipeline.named_steps["selector"].get_support().sum())
mlca_runtime.save_model(pipeline, "selector.pkl")
mlca_runtime.record_metric("n_features", float(len(input_cols)), "train")
mlca_runtime.record_metric("k", float(k), "train")
mlca_runtime.record_metric("n_selected", float(n_selected), "train")
print(f"train rows: {len(train)}, features: {len(input_cols)}, k: {k}, selected: {n_selected}")
