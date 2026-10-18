import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing '{key}'. Available top-level keys: {list(d.keys())}")
    return d[key]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group_col = need(protocol, "group_column")
task = need(protocol, "task")
_ = need(task, "primary_metric")

train = pd.read_parquet(inputs["train"])

with open(inputs["05_preprocess_fit_preprocessing_NBMSGCPN/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

exclude = set(targets) | set(excluded)
if group_col is not None:
    exclude.add(group_col)

candidate_cols = [c for c in train.columns if c not in exclude]
from pandas.api.types import is_object_dtype, is_string_dtype, is_integer_dtype, is_float_dtype

feature_cols = []
for c in candidate_cols:
    s = train[c]
    key_like = False
    if (is_object_dtype(s) or is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)):
        key_like = s.nunique(dropna=False) == len(train)
    elif is_integer_dtype(s) and not is_float_dtype(s):
        key_like = s.nunique(dropna=False) == len(train)
    if not key_like:
        feature_cols.append(c)

if hasattr(preprocessor, "feature_names_in_"):
    pcols = list(preprocessor.feature_names_in_)
    missing = [c for c in pcols if c not in feature_cols]
    if missing:
        raise ValueError(f"Preprocessor expects columns outside the allowed feature set: {missing}")
    feature_cols = pcols

X_train = train[feature_cols]
y_train = train[targets[0]]

if hasattr(preprocessor, "get_feature_names_out"):
    try:
        n_features = len(preprocessor.get_feature_names_out())
    except Exception:
        n_features = preprocessor.transform(X_train).shape[1]
else:
    n_features = preprocessor.transform(X_train).shape[1]

k = min(20, n_features)
steps = list(preprocessor.steps) if isinstance(preprocessor, Pipeline) else [("preprocessor", preprocessor)]
steps.append(("selector", SelectKBest(score_func=mutual_info_classif, k=k)))
pipeline = Pipeline(steps=steps)

fitted = mlca_runtime.fit_record(pipeline, X_train, "train", y_train)
mlca_runtime.save_model(fitted, "selector.pkl")

n_selected = int(fitted.named_steps["selector"].get_support().sum())
mlca_runtime.record_metric("n_features_after_preprocessing", int(n_features), "train")
mlca_runtime.record_metric("n_features_selected", n_selected, "train")
print(f"train rows={len(train)}, features={len(feature_cols)}, preprocessed={n_features}, k={k}, selected={n_selected}")
