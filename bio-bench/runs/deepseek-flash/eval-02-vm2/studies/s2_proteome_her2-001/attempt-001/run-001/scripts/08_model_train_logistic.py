import json
import pickle

import mlca_runtime
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

try:
    targets = proto["targets"]
    excluded_raw = proto["excluded_columns"]
    task = proto["task"]
except KeyError as exc:
    raise KeyError(f"Missing protocol key {exc}; available keys: {sorted(proto.keys())}") from exc

group = proto.get("group_column")
if isinstance(targets, str):
    targets = [targets]
if isinstance(excluded_raw, dict):
    excluded = set(excluded_raw.get("examples", []))
else:
    excluded = set(excluded_raw)
group_cols = set() if group is None else ({group} if isinstance(group, str) else set(group))

train = pd.read_parquet(inputs["train"])
target = targets[0]
bad_cols = set(targets) | excluded | group_cols
feature_cols = [c for c in train.columns if c not in bad_cols]

n = len(train)
for c in list(feature_cols):
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=False) == n:
            feature_cols.remove(c)
    elif pd.api.types.is_integer_dtype(s.dtype):
        if s.nunique(dropna=False) == n:
            feature_cols.remove(c)

with open(inputs["06_preprocess_fit_preprocessing_R9QMSZCE/preprocessor.pkl"], "rb") as f:
    preproc = pickle.load(f)
with open(inputs["07_features_select_features_C8HA72DE/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

def has_preprocessing(obj):
    if hasattr(obj, "transformers"):
        return True
    if isinstance(obj, Pipeline):
        return any(("preprocess" in name.lower()) or hasattr(step, "transformers")
                   for name, step in obj.named_steps.items())
    return False

sel_in = list(selector.feature_names_in_) if hasattr(selector, "feature_names_in_") else None
try:
    pre_out = list(preproc.get_feature_names_out())
except Exception:
    pre_out = None

logit = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

if has_preprocessing(selector):
    frontend_in = sel_in if sel_in is not None else feature_cols
    pipeline = Pipeline([("features", selector), ("model", logit)])
elif sel_in is not None and pre_out is not None and set(sel_in) == set(pre_out):
    frontend_in = list(preproc.feature_names_in_) if hasattr(preproc, "feature_names_in_") else feature_cols
    pipeline = Pipeline([("preprocessor", preproc), ("selector", selector), ("model", logit)])
elif sel_in is not None and set(sel_in) == set(feature_cols):
    frontend_in = sel_in
    pipeline = Pipeline([("features", selector), ("model", logit)])
else:
    frontend_in = list(preproc.feature_names_in_) if hasattr(preproc, "feature_names_in_") else feature_cols
    pipeline = Pipeline([("preprocessor", preproc), ("selector", selector), ("model", logit)])

feature_cols = list(frontend_in)
bad = set(feature_cols) & bad_cols
if bad:
    raise ValueError(f"Target/excluded/group columns reached the model: {sorted(bad)}")
missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise KeyError(f"Missing feature columns: {missing}")

pipeline = mlca_runtime.fit_record(pipeline, train[feature_cols], "train", y=train[target])
mlca_runtime.save_model(pipeline, "model.pkl")
print(f"train rows={len(train)} features={len(feature_cols)} target={target}")
print("saved model.pkl")
