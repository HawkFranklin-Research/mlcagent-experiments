import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_5G7E554W/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
try:
    group_col = protocol["group_column"]
except KeyError as exc:
    raise KeyError(f"Missing 'group_column'. Available keys: {list(protocol.keys())}") from exc

target = targets[0]
y = train[target]
drop = set(targets) | set(excluded)
if group_col is not None:
    drop.add(group_col)

def is_key_like(s, name):
    lname = name.lower()
    if any(t in lname for t in ("id", "key", "index")) and not pd.api.types.is_float_dtype(s):
        return True
    if s.nunique(dropna=False) == len(s):
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            return True
        if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
            return True
    return False

feature_cols = [c for c in train.columns if c not in drop and not is_key_like(train[c], c)]
if hasattr(preprocessor, "feature_names_in_"):
    expected = list(preprocessor.feature_names_in_)
    if set(expected) != set(feature_cols):
        raise ValueError(f"Preprocessor expects {expected}, derived {feature_cols}")
    feature_cols = expected

X = train[feature_cols]
try:
    X_pre = preprocessor.transform(X)
except NotFittedError:
    preprocessor = mlca_runtime.fit_record(preprocessor, X, "train", y)
    X_pre = preprocessor.transform(X)

n_features = X_pre.shape[1]
if n_features <= 300 and hasattr(preprocessor, "set_output"):
    try:
        preprocessor.set_output(transform="pandas")
        X_pre = preprocessor.transform(X)
    except Exception:
        pass

k = min(20, n_features)
selector = SelectKBest(score_func=mutual_info_classif, k=k)
selector = mlca_runtime.fit_record(selector, X_pre, "train", y)

final = Pipeline([("preprocessor", preprocessor), ("selector", selector)])
if n_features <= 300 and hasattr(selector, "set_output"):
    try:
        final.set_output(transform="pandas")
    except Exception:
        pass

mlca_runtime.save_model(final, "selector.pkl")
mlca_runtime.record_metric("n_features_preprocessed", float(n_features), "train")
mlca_runtime.record_metric("selector_k", float(k), "train")
print(f"train shape: {train.shape}")
print(f"preprocessed features: {n_features}; selected k: {k}")
