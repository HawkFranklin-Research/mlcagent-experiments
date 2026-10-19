import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in proto:
        raise KeyError(f"Missing protocol key {key}; available top-level keys: {list(proto)}")

targets = list(proto["targets"])
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]
excluded = list(proto["excluded_columns"] or [])
group_col = proto["group_column"]
metric = proto["task"]["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
with open(inp["05_preprocess_fit_preprocessing_MQJ4AQ8S/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_select_features_ZWFAMHT4/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
computed = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    n = len(s)
    keylike = False
    if n > 2:
        if s.dtype == object or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            keylike = s.nunique(dropna=False) == n
        elif pd.api.types.is_integer_dtype(s):
            vals = s.dropna().to_numpy()
            if len(vals) == n and len(np.unique(vals)) == n:
                u = np.sort(vals)
                if not (len(u) <= 2 and set(u).issubset({0, 1})):
                    keylike = np.array_equal(u, np.arange(u[0], u[0] + n)) and u[0] in (0, 1)
    if not keylike:
        computed.append(c)

def names_of(obj):
    fn = getattr(obj, "feature_names_in_", None)
    if fn is not None:
        return list(fn)
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            fn = getattr(step, "feature_names_in_", None)
            if fn is not None:
                return list(fn)
    return None

pre_names = names_of(pre)
sel_names = names_of(sel)
feature_cols = list(computed)
if pre_names is not None:
    if set(pre_names) != set(computed):
        raise ValueError(
            f"Preprocessor input mismatch; missing={set(pre_names) - set(computed)}, "
            f"extra={set(computed) - set(pre_names)}"
        )
    feature_cols = list(pre_names)

def selector_has_raw(obj, raw_cols):
    if not isinstance(obj, Pipeline):
        return False
    raw_set = set(raw_cols)
    for step in list(obj.named_steps.values())[:-1]:
        fn = getattr(step, "feature_names_in_", None)
        if fn is not None and set(fn) == raw_set:
            return True
    return False

sel_raw = selector_has_raw(sel, feature_cols)
if sel_raw:
    if sel_names is not None:
        if set(sel_names) != set(computed):
            raise ValueError(
                f"Selector input mismatch; missing={set(sel_names) - set(computed)}, "
                f"extra={set(computed) - set(sel_names)}"
            )
        feature_cols = list(sel_names)
    steps = [("selector", sel)]
else:
    if sel_names is not None:
        try:
            pre_out = list(pre.get_feature_names_out())
        except Exception:
            pre_out = None
        if pre_out is not None and list(sel_names) != pre_out:
            raise ValueError("Selector input names do not match preprocessor output names")
    steps = [("preprocessor", pre), ("selector", sel)]
steps.append(("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)))
pipeline = Pipeline(steps)

X_train = train[feature_cols]
y_train = train[target]
fitted = mlca_runtime.fit_record(pipeline, X_train, "train", y_train)
model = fitted if fitted is not None else pipeline
mlca_runtime.save_model(model, "model.pkl")

X_val = val[feature_cols]
y_val = val[target]
proba = model.predict_proba(X_val)
classes = model.classes_
y_arr = y_val.to_numpy()
present = [i for i, c in enumerate(classes) if np.any(y_arr == c)]
if len(present) == len(classes):
    score = roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)
else:
    score = float(np.mean([
        roc_auc_score((y_arr == classes[i]).astype(int), proba[:, i]) for i in present
    ]))
mlca_runtime.record_metric(metric, float(score), "val")
print(f"train={X_train.shape} val={X_val.shape} {metric}={score:.4f}")
