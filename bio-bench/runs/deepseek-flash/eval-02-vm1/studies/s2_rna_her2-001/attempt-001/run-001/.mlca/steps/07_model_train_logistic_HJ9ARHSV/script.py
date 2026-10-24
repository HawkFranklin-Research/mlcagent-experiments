import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"missing {k}; available top-level keys: {list(d.keys())}")
    return d[k]

targets = need(proto, "targets")
target = targets[0]
task = need(proto, "task")

train = pd.read_parquet(inputs["train"])
y = train[target]

with open(inputs["05_preprocess_fit_preprocessing_Y84CNP1M/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_TG41SQ76/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_cols(obj):
    cols = getattr(obj, "feature_names_in_", None)
    if cols is not None:
        return list(cols)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            c = getattr(step, "feature_names_in_", None)
            if c is not None:
                return list(c)
    return []

pre_cols = input_cols(pre)
sel_cols = input_cols(sel)
raw_cols = pre_cols or sel_cols

use_sel_only = bool(pre_cols and sel_cols and sel_cols == pre_cols)
clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

if use_sel_only:
    model = Pipeline([("select", sel), ("clf", clf)])
else:
    model = Pipeline([("preprocess", pre), ("select", sel), ("clf", clf)])

if not raw_cols:
    excluded = set(proto.get("excluded_columns", []))
    group = proto.get("group_column")
    raw_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
    kept = []
    for c in raw_cols:
        s = train[c]
        is_int = pd.api.types.is_integer_dtype(s.dtype)
        is_txt = s.dtype == object or str(s.dtype).startswith("string")
        all_distinct = s.nunique(dropna=False) == len(s)
        indicator = is_int and s.dropna().isin([0, 1]).all()
        if (is_txt or is_int) and all_distinct and not indicator:
            continue
        kept.append(c)
    raw_cols = kept

X = train[raw_cols].copy()
model = mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(model, "model.pkl")
print(f"train rows={len(train)} features={X.shape[1]}")
