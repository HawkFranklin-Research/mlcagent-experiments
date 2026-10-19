import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import poisson_deviance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in proto:
        raise KeyError(f"Missing {key!r}; available top-level keys: {list(proto)}")
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group_col = proto.get("group_column")
task = proto["task"]
primary_metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = [c for c in train.columns if c not in drop]
for c in list(feature_cols):
    s = train[c]
    if s.nunique(dropna=False) == len(train) and (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_integer_dtype(s)
    ):
        feature_cols.remove(c)

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inputs["05_preprocess_fit_preprocessing_SMPGAWB4/preprocessor.pkl"])
sel = load_pkl(inputs["06_features_retain_features_CANXQVVF/selector_all.pkl"])

sel_is_raw = (
    hasattr(sel, "feature_names_in_")
    and set(sel.feature_names_in_).issubset(set(train.columns))
)

steps = []
if sel_is_raw:
    steps.append(("selector", sel))
else:
    steps.append(("preprocess", pre))
    include_sel = True
    if hasattr(pre, "get_feature_names_out") and hasattr(sel, "feature_names_in_"):
        try:
            out_names = set(pre.get_feature_names_out())
            in_names = set(sel.feature_names_in_)
            include_sel = in_names.issubset(out_names) or in_names == out_names
        except Exception:
            include_sel = False
    if include_sel:
        steps.append(("select", sel))
steps.append(("model", LinearRegression()))
pipeline = Pipeline(steps)

X_train = train[feature_cols]
y_train = train[targets]
fitted = mlca_runtime.fit_record(pipeline, X_train, "train", y=y_train)

X_val = val[feature_cols]
y_val = val[targets]
pred = np.asarray(fitted.predict(X_val))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
pred = np.clip(pred, 1e-6, None)

scores = []
for j, target in enumerate(targets):
    true = y_val.iloc[:, j].to_numpy()
    score = float(poisson_deviance(true, pred[:, j]))
    scores.append(score)
    mlca_runtime.record_metric(f"{primary_metric}_{target}", score, "val")
mean_score = float(np.mean(scores))
mlca_runtime.record_metric(primary_metric, mean_score, "val")

mlca_runtime.save_model(fitted, "model.pkl")
print(f"train rows: {len(train)}, features: {len(feature_cols)}")
print(f"val {primary_metric}: {mean_score:.4f}")
