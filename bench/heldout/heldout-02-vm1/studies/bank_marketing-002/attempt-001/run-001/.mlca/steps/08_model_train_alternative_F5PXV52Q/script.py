import json, pickle
import mlca_runtime
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"missing {k!r}; top-level keys: {sorted(d)}")
    return d[k]

targets = need(proto, "targets")
excluded = list(need(proto, "excluded_columns") or [])
group = proto.get("group_column")
_ = need(proto, "task")

train = pd.read_parquet(inputs["train"])
y = train[targets[0]]

feature_cols = [c for c in train.columns
                if c not in targets and c not in excluded and c != group]
for c in list(feature_cols):
    s = train[c]
    is_key_like = s.nunique(dropna=False) == len(train) and (
        s.dtype == object or pd.api.types.is_integer_dtype(s))
    if is_key_like:
        feature_cols.remove(c)

X = train[feature_cols]

with open(inputs["07_model_train_logistic_NVNX9XZ7/model.pkl"], "rb") as f:
    saved = pickle.load(f)

minority = float(min(y.value_counts(normalize=True)))
cw = "balanced" if minority < 0.10 else None
hgb = HistGradientBoostingClassifier(random_state=42, class_weight=cw)

if isinstance(saved, Pipeline):
    new_steps = list(saved.steps[:-1]) + [("hist_gradient_boosting", hgb)]
    pipe = Pipeline(new_steps)
else:
    with open(inputs["05_preprocess_fit_preprocessing_VSWY5WH2/preprocessor.pkl"], "rb") as f:
        pre = pickle.load(f)
    with open(inputs["06_features_select_features_4XA2PFRM/selector.pkl"], "rb") as f:
        sel = pickle.load(f)
    pipe = Pipeline([("preprocessor", pre), ("selector", sel),
                     ("hist_gradient_boosting", hgb)])

first = pipe.steps[0][1]
cols = getattr(pipe, "feature_names_in_", None)
if cols is None:
    cols = getattr(first, "feature_names_in_", None)
if cols is not None:
    keep = [c for c in cols if c in X.columns]
    X = X[keep]

mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(pipe, "alternative_model.pkl")
print(f"fit rows={len(X)} features={X.shape[1]} steps={len(pipe.steps)}")
