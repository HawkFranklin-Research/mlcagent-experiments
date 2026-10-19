import json
import pickle

import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())


def required(key):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {list(protocol)}")
    return protocol[key]


targets = required("targets") or []
excluded = required("excluded_columns") or []
group_col = required("group_column")
task = required("task")
metric_name = task["primary_metric"]

train = pd.read_parquet(inputs["train"])


def is_key_like(series):
    n = len(series)
    if n <= 1:
        return False
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return series.nunique(dropna=False) == n
    if pd.api.types.is_integer_dtype(series.dtype):
        return series.nunique(dropna=False) == n and (
            series.is_monotonic_increasing or series.is_monotonic_decreasing
        )
    return False


drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = [
    c for c in train.columns
    if c not in drop and not is_key_like(train[c])
]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


preprocessor = load_pickle(inputs["05_preprocess_fit_preprocessing_W49DB36F/preprocessor.pkl"])
selector = load_pickle(inputs["06_features_retain_features_XVA92YG5/selector_all.pkl"])


def contract_order(obj, cols):
    if not hasattr(obj, "feature_names_in_"):
        return None
    names = list(obj.feature_names_in_)
    if len(names) == len(cols) and set(names) == set(cols):
        return names
    return None


front = None
front_cols = None
front_name = None
for name, obj in [("selector_all", selector), ("preprocessor", preprocessor)]:
    names = contract_order(obj, feature_cols)
    if names is not None:
        front, front_cols, front_name = obj, names, name
        break
if front is None:
    raise ValueError("No saved frontend accepts the declared feature columns")

X = train.loc[:, front_cols]
probe = front.transform(X.head(min(5, len(X))))
if hasattr(probe, "toarray"):
    probe = probe.toarray()
n_input_features = int(np.asarray(probe).shape[1])
n_components = min(8, n_input_features)
pca_name = "pca"
if isinstance(front, Pipeline):
    if pca_name in front.named_steps:
        pca_name = "pca_final"
    pipeline = Pipeline(
        steps=list(front.steps)
        + [(pca_name, PCA(n_components=n_components, random_state=42))]
    )
else:
    pipeline = Pipeline(
        [("frontend", front), (pca_name, PCA(n_components=n_components, random_state=42))]
    )

pipeline = mlca_runtime.fit_record(pipeline, X, partition="train")

frontend = pipeline[:-1]
encoded = frontend.transform(X)
if hasattr(encoded, "toarray"):
    encoded = encoded.toarray()
encoded = np.asarray(encoded)
pca = pipeline.steps[-1][1]
reconstructed = pca.inverse_transform(pca.transform(encoded))
mse = float(np.mean((encoded - reconstructed) ** 2))

mlca_runtime.record_metric(metric_name, mse, "train")
mlca_runtime.save_model(pipeline, "model.pkl")
print(f"train_shape={train.shape} features={len(feature_cols)} frontend={front_name}")
print(f"pca_components={n_components} {metric_name}_train_frontend_space={mse:.6g}")
