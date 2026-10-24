import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

seed = 42
mlca_runtime.set_seed(seed)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
target = targets[0]
metric = protocol["task"]["primary_metric"]

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
y_train = train[target]
y_val = val[target]

model_path = paths["08_model_train_logistic_DZS6CMQH/model.pkl"]
base = pickle.loads(model_path.read_bytes())

if isinstance(base, Pipeline) and base.named_steps:
    steps = list(base.named_steps.items())
    last_name = steps[-1][0]
    pipe = Pipeline(steps[:-1] + [(last_name, HistGradientBoostingClassifier(random_state=seed))])
else:
    pre = pickle.loads(paths["06_preprocess_fit_preprocessing_HTY1VTC0/preprocessor.pkl"].read_bytes())
    sel = pickle.loads(paths["07_features_select_features_Z274ACYD/selector.pkl"].read_bytes())
    last_name = "classifier"
    pipe = Pipeline([("preprocessor", pre), ("selector", sel),
                     (last_name, HistGradientBoostingClassifier(random_state=seed))])

if hasattr(base, "feature_names_in_"):
    feature_cols = [c for c in base.feature_names_in_
                    if c in train.columns and c not in targets and c not in excluded and c != group]
else:
    feature_cols = [c for c in train.columns
                    if c not in targets and c not in excluded and c != group]
    feature_cols = [c for c in feature_cols
                    if not ((pd.api.types.is_integer_dtype(train[c])
                             or pd.api.types.is_object_dtype(train[c])
                             or pd.api.types.is_string_dtype(train[c]))
                            and train[c].nunique(dropna=True) == len(train))]

if not feature_cols:
    raise ValueError(f"No feature columns found; protocol keys={list(protocol)}")

X_train = train.loc[:, feature_cols].copy()
X_val = val.loc[:, feature_cols].copy()
pipe = mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)

est = pipe.named_steps[last_name]
classes = est.classes_
proba = pipe.predict_proba(X_val)
present = np.array([c in set(y_val) for c in classes])
if present.all():
    score = roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)
else:
    score = float(np.mean([
        roc_auc_score(y_val == classes[j], proba[:, j])
        for j in range(len(classes)) if present[j]
    ]))
    print("val classes absent:", list(classes[~present]))

mlca_runtime.record_metric(metric, float(score), "val")
mlca_runtime.save_model(pipe, "alternative_model.pkl")

print("train rows", len(train), "features", len(feature_cols))
print("model", type(est).__name__)
print(metric, round(float(score), 4))
