import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(d, key):
    if key not in d:
        raise KeyError(f"missing {key}; available top-level keys: {list(d)}")
    return d[key]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
task = need(protocol, "task")
metric = need(task, "primary_metric")
target = targets[0]
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_key_like(s, name):
    lname = name.lower()
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=True) == len(s) and any(t in lname for t in ("id", "key", "code", "index"))
    if pd.api.types.is_integer_dtype(s):
        vals = s.dropna().unique()
        if len(vals) == len(s) and len(vals) > 1:
            lo, hi = int(vals.min()), int(vals.max())
            seq = np.arange(lo, hi + 1)
            if len(seq) == len(vals) and np.array_equal(np.sort(vals), seq):
                return any(t in lname for t in ("id", "key", "index", "code"))
    return False

drop_cols = set(targets) | set(excluded)
if group_col:
    drop_cols.add(group_col)
feature_cols = [c for c in train.columns if c not in drop_cols]
feature_cols = [c for c in feature_cols if not is_key_like(train[c], c)]

X_train, y_train = train[feature_cols], train[target]
X_val, y_val = val[feature_cols], val[target]

with open(inputs["06_preprocess_fit_preprocessing_2FS9KME6/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["07_features_retain_features_SY1WG08Y/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def maybe_pandas(obj):
    if hasattr(obj, "set_output"):
        try:
            obj.set_output(transform="pandas")
        except Exception:
            pass

def has_preprocessing(obj):
    if isinstance(obj, ColumnTransformer):
        return True
    if not isinstance(obj, Pipeline):
        return False
    steps = list(obj.named_steps.values())
    if not steps:
        return False
    first = steps[0]
    if isinstance(first, ColumnTransformer):
        return True
    names = [n.lower() for n in obj.named_steps]
    return any("preprocess" in n for n in names) or first.__class__.__name__.lower().endswith("preprocessor")

sel_has_pre = has_preprocessing(sel)
lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
if sel_has_pre:
    maybe_pandas(sel)
    pipe = Pipeline([("selector", sel), ("model", lr)])
else:
    maybe_pandas(pre)
    pipe = Pipeline([("preprocessor", pre), ("selector", sel), ("model", lr)])

fitted = mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "model.pkl")

proba = fitted.predict_proba(X_val)
classes = fitted.named_steps["model"].classes_
present = [c for c in classes if (y_val == c).any()]
if len(present) == len(classes):
    score = roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)
else:
    score = float(np.mean([
        roc_auc_score((y_val == c).astype(int), proba[:, list(classes).index(c)])
        for c in present
    ]))
    print(f"val classes missing: {len(classes) - len(present)}")

mlca_runtime.record_metric("val_roc_auc_ovr", float(score), "val")
print(f"target={target} metric={metric}")
print(f"train rows={len(X_train)} features={len(feature_cols)}")
print(f"selector_has_preprocessing={sel_has_pre}")
print(f"val roc_auc_ovr={score:.4f}")
