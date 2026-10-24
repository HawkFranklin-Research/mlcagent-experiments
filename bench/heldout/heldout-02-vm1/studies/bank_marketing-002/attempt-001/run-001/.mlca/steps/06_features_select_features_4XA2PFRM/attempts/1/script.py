import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
train = pd.read_parquet(paths["train"])
protocol = json.loads(paths["_protocol"].read_text())
with open(paths["05_preprocess_fit_preprocessing_VSWY5WH2/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)


def is_key_like(s):
    if s.isna().any() or s.nunique(dropna=True) != len(s):
        return False
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    return pd.api.types.is_integer_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype)


targets = list(protocol["targets"])
target = targets[0]
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
drop = set(targets) | excluded | ({group} if group else set())

if hasattr(pre, "feature_names_in_"):
    feature_cols = list(pre.feature_names_in_)
    missing = [c for c in feature_cols if c not in train.columns]
    if missing:
        raise ValueError(f"Preprocessor expects missing columns: {missing}")
    if any(c in drop for c in feature_cols):
        raise ValueError("Preprocessor input includes a target, excluded or group column")
    if any(is_key_like(train[c]) for c in feature_cols):
        raise ValueError("Preprocessor input includes a key-like column")
else:
    feature_cols = [c for c in train.columns if c not in drop and not is_key_like(train[c])]

X_train = train.loc[:, feature_cols]
y_train = LabelEncoder().fit_transform(train[target])

k_value = None
if isinstance(pre, Pipeline) and pre.steps and isinstance(pre.steps[-1][1], SelectKBest):
    pipeline = pre
else:
    n_features = None
    if hasattr(pre, "get_feature_names_out"):
        try:
            n_features = len(pre.get_feature_names_out())
        except Exception:
            pass
    k_value = max(1, min(20, n_features)) if n_features is not None else 20
    selector = SelectKBest(score_func=mutual_info_classif, k=k_value)
    if isinstance(pre, Pipeline):
        sel_name = "selector" if "selector" not in pre.named_steps else "selector_final"
        pipeline = Pipeline(steps=list(pre.steps) + [(sel_name, selector)])
    else:
        pipeline = Pipeline([("preprocess", pre), ("selector", selector)])

fitted = mlca_runtime.fit_record(pipeline, X_train, "train", y=y_train)
selected = None
for _, step in getattr(fitted, "steps", []):
    if hasattr(step, "get_support"):
        selected = int(step.get_support().sum())
if selected is None:
    selected = len(feature_cols)

mlca_runtime.record_metric("n_input_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_selected_features", float(selected), "train")
mlca_runtime.save_model(fitted, "selector.pkl")
print(f"input_features={len(feature_cols)} k={k_value} selected={selected}")
