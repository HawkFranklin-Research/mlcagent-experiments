import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)

inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())

targets = list(proto["targets"])
excluded = set(proto["excluded_columns"])
group = proto.get("group_column")
if group:
    excluded.add(group)

train = pd.read_parquet(inp["train"])
if len(targets) != 1:
    raise ValueError("expected exactly one target")
target = targets[0]
if target not in train.columns:
    raise ValueError(f"target column {target!r} is missing")
y = train[target]

with open(inp["05_preprocess_fit_preprocessing_B1ZJVKDM/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

def raw_feature_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "named_steps"):
        for _, step in obj.steps:
            names = raw_feature_names(step)
            if names is not None:
                return names
    return None

expected = raw_feature_names(pre)
if expected is not None:
    bad = [c for c in expected if c in excluded or c == target]
    if bad:
        raise ValueError(f"preprocessor expects excluded/target columns: {bad[:5]}")
    feature_cols = expected
else:
    feature_cols = []
    for c in train.columns:
        if c in excluded or c == target:
            continue
        s = train[c]
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
            continue
        if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s) and (c.lower().endswith("id") or c.lower().endswith("code")):
            continue
        feature_cols.append(c)

missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise ValueError(f"feature columns missing from train: {missing[:5]}")

X = train[feature_cols]
X_trans = pre.transform(X)
n_features = X_trans.shape[1]

if n_features <= 300 and not isinstance(X_trans, pd.DataFrame):
    try:
        pre.set_output(transform="pandas")
        X_trans = pre.transform(X)
        n_features = X_trans.shape[1]
    except Exception:
        pass

k = min(20, n_features)
if k < 1:
    raise ValueError("no features available for selection")

selector = SelectKBest(score_func=mutual_info_classif, k=k)
pipe = Pipeline([("preprocessor", pre), ("selector", selector)])
mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(pipe, "selector.pkl")

mlca_runtime.record_metric("n_features_input", float(n_features), "train")
mlca_runtime.record_metric("n_features_selected", float(k), "train")
print(f"input features: {n_features}")
print(f"selected features: {k}")
