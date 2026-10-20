import mlca_runtime
import json
import pickle
from functools import partial

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol)}")

targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
train = pd.read_parquet(inputs["train"])

feature_columns = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    key_like = (
        s.nunique(dropna=False) == len(train)
        and (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
             or isinstance(s.dtype, pd.CategoricalDtype)
             or pd.api.types.is_integer_dtype(s))
        and not pd.api.types.is_bool_dtype(s)
    )
    if not key_like:
        feature_columns.append(c)

with open(inputs["05_preprocess_fit_preprocessing_6KJVP4ST/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    expected = list(pre.feature_names_in_)
    bad = [c for c in expected if c in targets or c in excluded or c == group]
    if bad:
        raise ValueError(f"Preprocessor expects forbidden columns: {bad}")
    feature_columns = expected

try:
    n_features = len(pre.get_feature_names_out())
except Exception:
    n_features = pre.transform(train[feature_columns].head(1)).shape[1]
if n_features < 1:
    raise ValueError("Preprocessor produced no features")

k = min(20, n_features)
has_selector = isinstance(pre, Pipeline) and any(
    isinstance(step, SelectKBest) for step in pre.named_steps.values()
)
if has_selector:
    selector = pre
else:
    selector = Pipeline([
        ("preprocess", pre),
        ("select", SelectKBest(
            score_func=partial(mutual_info_classif, random_state=42),
            k=k,
        )),
    ])

X = train[feature_columns]
y = train[targets[0]]
mlca_runtime.fit_record(selector, X, "train", y=y)
mlca_runtime.save_model(selector, "selector.pkl")

mlca_runtime.record_metric("n_train_rows", len(train), "train")
mlca_runtime.record_metric("n_features_in", len(feature_columns), "train")
mlca_runtime.record_metric("n_preprocessed_features", n_features, "train")
if not has_selector:
    mlca_runtime.record_metric("k_selected", k, "train")
print(f"rows={len(train)} features_in={len(feature_columns)} preprocessed={n_features}")
print(f"selector={'existing' if has_selector else f'SelectKBest(k={k})'}")
