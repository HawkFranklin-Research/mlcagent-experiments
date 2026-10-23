import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns") or [])
group = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(inputs["train"])
y = train[target]

with open(inputs["06_preprocess_fit_preprocessing_FTEQSYVR/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["07_features_retain_features_C5JWX6VS/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_cols(obj):
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

pre_cols = input_cols(pre)
sel_cols = input_cols(sel)

if pre_cols is not None:
    feature_cols = [c for c in pre_cols if c in train.columns]
else:
    feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]

    def key_like(s):
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            return s.nunique(dropna=True) == len(s)
        if pd.api.types.is_integer_dtype(s):
            return s.nunique(dropna=True) == len(s)
        return False

    feature_cols = [c for c in feature_cols if not key_like(train[c])]

for bad in set(targets) | set(excluded):
    if bad in feature_cols:
        raise ValueError(f"Outcome/excluded column {bad!r} would enter features")
if group is not None and group in feature_cols:
    raise ValueError(f"Group column {group!r} would enter features")

X = train[feature_cols].copy()

if sel is not None and sel_cols is not None and pre_cols is not None and set(sel_cols) == set(pre_cols):
    steps = [("selector_pipeline", sel)]
else:
    steps = []
    if pre is not None:
        steps.append(("preprocessor", pre))
    if sel is not None:
        steps.append(("selector", sel))

steps.append(("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)))
pipe = Pipeline(steps)
pipe = mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(pipe, "model.pkl")

print(f"train rows={len(X)} features={X.shape[1]}")
print(f"steps={[name for name, _ in pipe.steps]}")
