import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

def load_pkl(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("05_preprocess_fit_preprocessing_F5B5FT5N/preprocessor.pkl")
sel = load_pkl("06_features_select_features_ZZ9TRA69/selector.pkl")
proto = json.loads(inp["_protocol"].read_text())

targets = proto["targets"]
target = targets[0]
excluded = set(proto["excluded_columns"])
group = proto.get("group_column")
forbidden = set(targets) | excluded | ({group} if group else set())

train = pd.read_parquet(inp["train"])
y = train[target]

raw_names = getattr(pre, "feature_names_in_", None)
if raw_names is not None:
    missing = [c for c in raw_names if c not in train.columns]
    if missing:
        raise ValueError(f"missing preprocessor input columns: {missing}")
    feature_cols = list(raw_names)
else:
    base = [c for c in train.columns if c not in forbidden]
    feature_cols = []
    for c in base:
        s = train[c]
        n = s.nunique(dropna=True)
        is_key = n == len(s) and (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_integer_dtype(s)
        )
        if not is_key:
            feature_cols.append(c)

bad = forbidden.intersection(feature_cols)
if bad:
    raise ValueError(f"forbidden columns in features: {sorted(bad)}")

X = train[feature_cols].copy()

def starts_raw(obj):
    if isinstance(obj, Pipeline):
        if not obj.named_steps:
            return False
        obj = next(iter(obj.named_steps.values()))
    names = getattr(obj, "feature_names_in_", None)
    return names is not None and list(names) == feature_cols

model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

if starts_raw(sel):
    if isinstance(sel, Pipeline) and len(sel.named_steps) > 1:
        pipe = Pipeline([("selector_pipeline", sel), ("model", model)])
    else:
        pipe = Pipeline([
            ("union", FeatureUnion([("preprocess", pre), ("selector", sel)])),
            ("model", model),
        ])
else:
    pipe = Pipeline([("preprocess", pre), ("selector", sel), ("model", model)])

fitted = mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(fitted, "model.pkl")
print(f"train rows={len(X)} features={len(feature_cols)}")
print(f"saved model.pkl steps={len(fitted.named_steps)}")
