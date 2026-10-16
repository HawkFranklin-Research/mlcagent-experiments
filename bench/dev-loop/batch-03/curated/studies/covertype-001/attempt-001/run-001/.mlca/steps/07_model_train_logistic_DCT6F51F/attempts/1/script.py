import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing key {k!r} in _protocol. Available top-level keys: {list(d.keys())}")
    return d[k]

targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
group_col = req(protocol, "group_column")
task = req(protocol, "task")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
target = targets[0]
y_train = train[target]
y_val = val[target]

all_cols = list(train.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]

def is_key_like(s):
    if s.isna().any():
        return False
    if s.nunique(dropna=False) != len(s):
        return False
    if pd.api.types.is_float_dtype(s):
        return False
    return True

feature_cols = [c for c in feature_cols if not is_key_like(train[c])]

pre = pickle.load(open(inputs["05_preprocess_fit_preprocessing_1A6AJYGR/preprocessor.pkl"], "rb"))
sel = pickle.load(open(inputs["06_features_select_features_JT24GMYZ/selector.pkl"], "rb"))

if hasattr(pre, "feature_names_in_"):
    pre_cols = [c for c in pre.feature_names_in_ if c in feature_cols]
    if pre_cols:
        feature_cols = pre_cols

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()

def wraps_preprocessing(sel, pre):
    if not isinstance(sel, Pipeline):
        return False
    for name, _ in sel.steps:
        if "preprocess" in name.lower():
            return True
    first = sel.steps[0][1]
    if isinstance(first, (ColumnTransformer, Pipeline)):
        return True
    if hasattr(first, "feature_names_in_") and hasattr(pre, "feature_names_in_"):
        return list(first.feature_names_in_) == list(pre.feature_names_in_)
    return False

lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
if wraps_preprocessing(sel, pre):
    pipeline = Pipeline([("selector", sel), ("model", lr)])
else:
    pipeline = Pipeline([("preprocessor", pre), ("selector", sel), ("model", lr)])

model = mlca_runtime.fit_record(pipeline, X_train, partition="train", y=y_train)

proba = model.predict_proba(X_val)
classes = model.steps[-1][1].classes_
auc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)
mlca_runtime.record_metric("roc_auc_ovr", float(auc), split="val")
mlca_runtime.save_model(model, "model.pkl")

print(f"train shape={X_train.shape}, val shape={X_val.shape}")
print(f"features={len(feature_cols)}, classes={len(classes)}")
print(f"val roc_auc_ovr={auc:.4f}")
