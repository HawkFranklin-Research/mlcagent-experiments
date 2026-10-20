import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SEED = 42
mlca_runtime.set_seed(SEED)

inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]

def is_key_like(s):
    if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        vals = set(s.dropna().unique())
        if vals.issubset({0, 1}):
            return False
        return s.nunique(dropna=False) == len(s)
    return False

feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group
]
feature_cols = [c for c in feature_cols if not is_key_like(train[c])]

def input_names(obj):
    names = getattr(obj, "feature_names_in_", None)
    return list(names) if names is not None else None

base = pickle.load(open(inp["latest_model"], "rb"))
hgb = HistGradientBoostingClassifier(random_state=SEED)

if isinstance(base, Pipeline):
    steps = list(base.steps)
    last_name = steps[-1][0]
    raw_in = input_names(base) or feature_cols
    steps[-1] = (last_name, hgb)
    pipe = Pipeline(steps)
else:
    pre = pickle.load(open(inp["06_preprocess_fit_preprocessing_2FS9KME6/preprocessor.pkl"], "rb"))
    sel = pickle.load(open(inp["07_features_retain_features_SY1WG08Y/selector_all.pkl"], "rb"))
    sel_in = input_names(sel) if isinstance(sel, Pipeline) else None
    if sel_in is not None and set(feature_cols).issubset(set(sel_in)):
        raw_in = sel_in
        pipe = Pipeline([("features", sel), ("model", hgb)])
        last_name = "model"
    else:
        raw_in = input_names(pre) or feature_cols
        pipe = Pipeline([("preprocessor", pre), ("selector", sel), ("model", hgb)])
        last_name = "model"

raw_in = [c for c in raw_in if c in train.columns]
for bad in set(targets + excluded):
    if bad in raw_in:
        raise ValueError(f"Leakage column in model input: {bad}")

X_train = train.loc[:, raw_in].copy()
y_train = train[target].copy()
pipe = mlca_runtime.fit_record(pipe, X_train, "train", y_train)
mlca_runtime.save_model(pipe, "alternative_model.pkl")

X_val = val.loc[:, raw_in].copy()
y_val = val[target]
proba = pipe.predict_proba(X_val)
classes = list(pipe.named_steps[last_name].classes_)
present = [c for c in classes if (y_val == c).any()]
if len(present) == len(classes):
    auc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)
else:
    vals = []
    for c in present:
        j = classes.index(c)
        vals.append(roc_auc_score((y_val == c).astype(int), proba[:, j]))
    auc = float(np.mean(vals))
    print("missing classes", sorted(set(classes) - set(present)))

mlca_runtime.record_metric(protocol["task"]["primary_metric"], float(auc), "val")
print("features", len(raw_in), "train_rows", len(train))
print("val_auc", round(float(auc), 4))
