import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

import mlca_runtime

mlca_runtime.set_seed(42)
seed = 42

inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
protocol = json.loads(inp["_protocol"].read_text())

targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
target = targets[0]


def key_like(s: pd.Series) -> bool:
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return bool(s.notna().all() and s.nunique(dropna=True) == len(s))
    if pd.api.types.is_integer_dtype(s):
        return bool(s.notna().all() and s.nunique(dropna=True) == len(s))
    return False


feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    if key_like(train[c]):
        continue
    feature_cols.append(c)

X = train[feature_cols]
y = train[target]

with open(inp["05_preprocess_fit_preprocessing_8XQPADGG/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_select_features_NKRWNGEQ/selector.pkl"], "rb") as f:
    sel = pickle.load(f)


def raw_names(obj):
    names = getattr(obj, "feature_names_in_", None)
    return list(names) if names is not None else None


pre_names = raw_names(pre)
sel_names = raw_names(sel)

model = HistGradientBoostingClassifier(random_state=seed)
counts = y.value_counts(normalize=True)
if len(counts) and counts.min() < 0.10:
    try:
        model.set_params(class_weight="balanced")
    except TypeError:
        pass

if sel_names is not None and sel_names == feature_cols:
    steps = [("select", sel), ("model", model)]
elif pre_names is not None and pre_names == feature_cols:
    steps = [("preprocess", pre)]
    if sel is not None:
        pre_out = None
        try:
            pre_out = list(pre.get_feature_names_out())
        except Exception:
            pass
        if sel_names is None or (pre_out is not None and sel_names == pre_out):
            steps.append(("select", sel))
        elif sel_names == feature_cols:
            steps = [("select", sel)]
        else:
            raise ValueError("selector input does not match preprocessor output")
    steps.append(("model", model))
else:
    if pre is not None:
        steps.append(("preprocess", pre))
    if sel is not None:
        steps.append(("select", sel))
    steps.append(("model", model))

pipe = Pipeline(steps)
pipe = mlca_runtime.fit_record(pipe, X, "train", y=y)

proba = pipe.predict_proba(val[feature_cols])
classes = list(pipe.named_steps["model"].classes_)
y_val = val[target]
present = sorted(set(y_val.dropna().unique()))
scores = []
for c in present:
    yt = (y_val == c).to_numpy()
    if len(np.unique(yt)) < 2:
        continue
    scores.append(roc_auc_score(yt, proba[:, classes.index(c)]))

macro = float(np.mean(scores)) if scores else float("nan")
if not np.isnan(macro):
    mlca_runtime.record_metric("roc_auc_ovr", macro, "val")

missing = [c for c in classes if c not in present]
mlca_runtime.save_model(pipe, "alternative_model.pkl")
print(f"train={train.shape} features={len(feature_cols)}")
print(f"val_roc_auc_ovr={macro:.4f} missing_classes={missing}")
