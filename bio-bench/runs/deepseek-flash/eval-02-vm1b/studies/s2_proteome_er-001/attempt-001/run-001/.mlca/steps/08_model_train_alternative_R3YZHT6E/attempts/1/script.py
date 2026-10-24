import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

seed = 42
mlca_runtime.set_seed(seed)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
task = protocol["task"]
primary_metric = task["primary_metric"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
target = targets[0]

with open(inputs["latest_model"], "rb") as f:
    model_obj = pickle.load(f)

y_train = train[target]
minority = y_train.value_counts(normalize=True).min() if y_train.nunique() > 1 else 0.0
hgb = HistGradientBoostingClassifier(random_state=seed)
if minority < 0.10:
    hgb.set_params(class_weight="balanced")

feature_cols = None
if isinstance(model_obj, Pipeline):
    feature_cols = getattr(model_obj, "feature_names_in_", None)
    if feature_cols is None and len(model_obj.steps) > 0:
        feature_cols = getattr(model_obj.steps[0][1], "feature_names_in_", None)
    base = clone(model_obj)
    last_name = base.steps[-1][0]
    base.set_params(**{last_name: hgb})
else:
    with open(inputs["05_preprocess_fit_preprocessing_5K21TQY3/preprocessor.pkl"], "rb") as f:
        pre = pickle.load(f)
    with open(inputs["06_features_select_features_RDJRDVNY/selector.pkl"], "rb") as f:
        sel = pickle.load(f)
    feature_cols = getattr(pre, "feature_names_in_", None)
    base = Pipeline([
        ("preprocessor", clone(pre)),
        ("selector", clone(sel)),
        ("classifier", hgb),
    ])

if feature_cols is None or len(feature_cols) == 0:
    exclude = set(targets) | set(excluded) | ({group} if group else set())
    feature_cols = [c for c in train.columns if c not in exclude]
feature_cols = list(feature_cols)

X_train = train.loc[:, feature_cols]
X_val = val.loc[:, feature_cols]

fitted = mlca_runtime.fit_record(base, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "alternative_model.pkl")

auc = None
try:
    classes = list(fitted.classes_)
    pos = task.get("positive_label", classes[1] if len(classes) > 1 else classes[0])
    if pos not in classes:
        pos = classes[1] if len(classes) > 1 else classes[0]
    idx = classes.index(pos)
    y_val = val[target]
    y_true = (y_val == pos).astype(int)
    if y_true.nunique() > 1:
        proba = fitted.predict_proba(X_val)[:, idx]
        auc = roc_auc_score(y_true, proba)
        mlca_runtime.record_metric(primary_metric, float(auc), "val")
except Exception:
    auc = None

print(f"train={train.shape} val={val.shape}")
print(f"features={len(feature_cols)}")
print(f"val_{primary_metric}={auc if auc is None else round(float(auc), 4)}")
