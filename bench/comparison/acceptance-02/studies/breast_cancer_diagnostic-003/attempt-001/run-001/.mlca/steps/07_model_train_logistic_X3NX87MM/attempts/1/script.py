import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)


def required(d, key):
    if key not in d:
        raise KeyError(f"Missing {key!r}; available top-level keys: {sorted(d)}")
    return d[key]


def input_names(obj):
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        try:
            return list(obj.feature_names_in_)
        except Exception:
            pass
    if isinstance(obj, Pipeline) and obj.steps:
        return input_names(obj.steps[0][1])
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            return input_names(step)
    return None


def output_names(obj):
    if obj is None or not hasattr(obj, "get_feature_names_out"):
        return None
    try:
        return list(obj.get_feature_names_out())
    except Exception:
        return None


def is_key_like(s):
    n = len(s)
    if n == 0:
        return False
    name = str(s.name).lower()
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return bool(s.nunique(dropna=False) == n)
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        vals = s.dropna()
        if len(vals) != n or vals.nunique() != n:
            return False
        mn, mx = vals.min(), vals.max()
        return bool(name in {"id", "index"} or name.endswith("id") or (mn == 0 and mx == n - 1) or (mn == 1 and mx == n))
    return False


inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = required(protocol, "targets")
if isinstance(targets, str):
    targets = [targets]
excluded = required(protocol, "excluded_columns")
if isinstance(excluded, str):
    excluded = [excluded]
group_col = protocol.get("group_column")
task = required(protocol, "task")
primary_metric = task.get("primary_metric")

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_AKNK2K5H/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_HGTRMBKM/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

for c in targets:
    if c not in train.columns:
        raise KeyError(f"Target column {c!r} missing from train")

forbidden = set(targets) | set(excluded) | ({group_col} if group_col else set())
candidates = [c for c in train.columns if c not in forbidden and not is_key_like(train[c])]

pre_in = input_names(pre)
if pre_in is not None:
    missing = [c for c in pre_in if c not in train.columns]
    if missing:
        raise KeyError(f"Preprocessor expects missing columns: {missing}")
    bad = [c for c in pre_in if c in forbidden or is_key_like(train[c])]
    if bad:
        raise ValueError(f"Preprocessor expects forbidden columns: {bad}")
    feature_cols = list(pre_in)
else:
    feature_cols = candidates

if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")

X = train[feature_cols].copy()
y = train[targets[0]]

lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
sel_in = input_names(selector)

if sel_in is not None and set(sel_in) == set(feature_cols):
    model = Pipeline([("selector", selector), ("model", lr)])
else:
    pre_out = output_names(pre)
    if sel_in is not None and pre_out is not None and list(sel_in) != pre_out:
        raise ValueError("Selector input names do not match preprocessor output names")
    model = Pipeline([("preprocessing", pre), ("selector", selector), ("model", lr)])

fitted = mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(fitted, "model.pkl")
print(f"train rows={len(X)} features={len(feature_cols)}")
print(f"steps={list(fitted.named_steps)}")
print("saved model.pkl")
