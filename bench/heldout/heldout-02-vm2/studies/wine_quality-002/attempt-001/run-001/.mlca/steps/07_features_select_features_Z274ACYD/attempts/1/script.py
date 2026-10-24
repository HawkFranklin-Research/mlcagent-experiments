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

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
try:
    group = protocol["group_column"]
except KeyError as exc:
    raise KeyError(f"Protocol missing group_column; available keys: {sorted(protocol)}") from exc

train = pd.read_parquet(paths["train"])
if len(targets) != 1:
    raise ValueError(f"Expected exactly one target, got {targets}")
target = targets[0]

with open(paths["06_preprocess_fit_preprocessing_HTY1VTC0/preprocessor.pkl"], "rb") as fh:
    pre = pickle.load(fh)

forbidden = set(targets) | set(excluded) | ({group} if group is not None else set())

if hasattr(pre, "feature_names_in_"):
    feature_cols = list(pre.feature_names_in_)
else:
    feature_cols = [c for c in train.columns if c not in forbidden]
    feature_cols = [
        c for c in feature_cols
        if not (
            (pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]))
            and train[c].nunique(dropna=True) == len(train)
        )
    ]

missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise KeyError(f"Saved preprocessor expects columns absent from train: {missing}")
bad = forbidden.intersection(feature_cols)
if bad:
    raise ValueError(f"Target/excluded/group columns reached features: {sorted(bad)}")

X = train[list(feature_cols)]
y = train[target]

try:
    n_out = len(pre.get_feature_names_out())
except Exception:
    n_out = len(feature_cols)
k = min(20, n_out)

selector = SelectKBest(partial(mutual_info_classif, random_state=42), k=k)
pipeline = Pipeline([("preprocessor", pre), ("selector", selector)])
fitted = mlca_runtime.fit_record(pipeline, X, "train", y)

selected_n = int(fitted.named_steps["selector"].get_support().sum())
mlca_runtime.record_metric("n_selected_features", float(selected_n), "train")
mlca_runtime.save_model(fitted, "selector.pkl")

print(f"features_in={X.shape[1]} transformed={n_out} k={k} selected={selected_n} rows={len(X)}")
print("saved selector.pkl")
