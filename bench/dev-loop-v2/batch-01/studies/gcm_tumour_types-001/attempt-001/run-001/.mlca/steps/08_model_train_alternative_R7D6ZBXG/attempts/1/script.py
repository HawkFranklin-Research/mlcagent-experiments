import json
import pickle

import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

seed = 42
mlca_runtime.set_seed(seed)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
task = protocol["task"]
primary = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
target = targets[0]

with open(inputs["07_model_train_logistic_FX65AH12/model.pkl"], "rb") as f:
    saved_model = pickle.load(f)


def raw_feature_cols(df):
    cols, n = [], len(df)
    for c in df.columns:
        if c in targets or c in excluded or c == group_col:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            if pd.api.types.is_integer_dtype(s) and n > 1 and s.nunique(dropna=True) == n:
                continue
            cols.append(c)
        else:
            if s.nunique(dropna=True) > 0.05 * n:
                continue
            cols.append(c)
    return cols


if hasattr(saved_model, "feature_names_in_"):
    feature_cols = [c for c in saved_model.feature_names_in_ if c in train.columns]
else:
    feature_cols = raw_feature_cols(train)

missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise ValueError(f"missing feature columns: {missing[:5]}")

X = train[feature_cols]
y = train[target]

if isinstance(saved_model, Pipeline) and len(saved_model.steps) >= 2:
    pipe = clone(saved_model)
    pipe.steps[-1] = (pipe.steps[-1][0], HistGradientBoostingClassifier(random_state=seed))
else:
    pipe = Pipeline([("clf", HistGradientBoostingClassifier(random_state=seed))])

if y.value_counts(normalize=True).min() < 0.10:
    pipe.set_params(**{pipe.steps[-1][0] + "__class_weight": "balanced"})

fitted = mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(fitted, "alternative_model.pkl")

classes = getattr(fitted, "classes_", fitted.steps[-1][1].classes_)
proba = fitted.predict_proba(X)
auc = roc_auc_score(y.values, proba, multi_class="ovr", average="macro", labels=classes)
mlca_runtime.record_metric("train_roc_auc_ovr", float(auc), "train")

print("n_train", len(train), "n_features", len(feature_cols), "classes", int(y.nunique()))
print("train_roc_auc_ovr", round(float(auc), 4))
