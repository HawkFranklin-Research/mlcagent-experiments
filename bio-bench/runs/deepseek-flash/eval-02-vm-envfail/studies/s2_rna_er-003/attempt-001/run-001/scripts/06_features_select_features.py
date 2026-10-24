import mlca_runtime
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)

paths = mlca_runtime.inputs()
train = pd.read_parquet(paths["train"])
protocol = json.loads(Path(paths["_protocol"]).read_text())

with open(paths["05_preprocess_fit_preprocessing_K9GV8476/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if "targets" not in protocol:
    raise ValueError(f"Missing targets; available top-level keys: {sorted(protocol)}")
targets = list(protocol["targets"])
if not targets:
    raise ValueError("No targets declared in _protocol['targets']")
y_name = targets[0]

if "excluded_columns" not in protocol:
    raise ValueError(f"Missing excluded_columns; available top-level keys: {sorted(protocol)}")
excluded_raw = protocol["excluded_columns"]
if isinstance(excluded_raw, dict):
    if "columns" in excluded_raw:
        excluded = set(excluded_raw["columns"])
    elif "examples" in excluded_raw:
        excluded = set(excluded_raw["examples"])
    else:
        raise ValueError(f"excluded_columns dict lacks columns/examples; keys: {sorted(excluded_raw)}")
else:
    excluded = set(excluded_raw)

if "group_column" not in protocol:
    raise ValueError(f"Missing group_column; available top-level keys: {sorted(protocol)}")
group = protocol["group_column"]
forbidden = set(targets) | excluded | ({group} if group is not None else set())

if hasattr(pre, "feature_names_in_"):
    pre_features = list(pre.feature_names_in_)
    bad = sorted(set(pre_features) & forbidden)
    if bad:
        raise ValueError(f"Preprocessor input includes forbidden columns: {bad[:5]}")
    feature_cols = pre_features
else:
    feature_cols = [c for c in train.columns if c not in forbidden]
    n_rows = len(train)
    keep = []
    for c in feature_cols:
        s = train[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
            if s.nunique(dropna=False) == n_rows:
                continue
        elif pd.api.types.is_integer_dtype(s.dtype):
            vals = set(s.dropna().unique())
            if s.nunique(dropna=False) == n_rows and not vals.issubset({0, 1}):
                continue
        keep.append(c)
    feature_cols = keep

if not feature_cols:
    raise ValueError("No feature columns remain after exclusions")

X = train[feature_cols]
y = train[y_name]

def output_width(preprocessor, X_sample):
    try:
        return len(preprocessor.get_feature_names_out())
    except Exception:
        try:
            return preprocessor.transform(X_sample.iloc[:5]).shape[1]
        except Exception:
            return None

n_out = output_width(pre, X)
if n_out is None:
    n_out = len(feature_cols)
k = min(20, int(n_out))
if k < 1:
    raise ValueError("SelectKBest k must be >= 1")

pipeline = Pipeline([
    ("preprocessor", pre),
    ("selector", SelectKBest(score_func=mutual_info_classif, k=k)),
])

mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(pipeline, "selector.pkl")

print(f"train rows={len(train)} feature_cols={len(feature_cols)} pre_out={n_out}")
print(f"SelectKBest k={k} saved selector.pkl")
