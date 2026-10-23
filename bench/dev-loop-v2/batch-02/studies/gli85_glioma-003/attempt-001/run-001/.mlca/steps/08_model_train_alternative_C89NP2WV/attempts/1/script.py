import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
task = protocol["task"]
target = protocol["targets"][0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
with open(inputs["latest_model"], "rb") as f:
    base = pickle.load(f)

print("base:", type(base).__name__, "steps:", list(getattr(base, "named_steps", {}).keys()))

feature_cols = list(base.feature_names_in_)
X_train = train[feature_cols]; y_train = train[target]
X_val = val[feature_cols]; y_val = val[target]

counts = y_train.value_counts(normalize=True)
hgb_kwargs = dict(random_state=42)
if counts.min() < 0.10:
    hgb_kwargs["class_weight"] = "balanced"

pipe = clone(base)
last = list(pipe.named_steps)[-1]
pipe.set_params(**{last: HistGradientBoostingClassifier(**hgb_kwargs)})

fitted = mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)

classes = list(fitted.classes_)
proba = fitted.predict_proba(X_val)
pos = 1
y_bin = (y_val == classes[pos]).astype(int)
auc = roc_auc_score(y_bin, proba[:, pos])
mlca_runtime.record_metric("val_roc_auc", float(auc), "val")
print("n_features:", len(feature_cols), "smallest pct:", round(counts.min(), 3), "val auc:", round(auc, 4))

mlca_runtime.save_model(fitted, "alternative_model.pkl")
