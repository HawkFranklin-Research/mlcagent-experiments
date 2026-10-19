import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")
task = protocol["task"]
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
y_train = train[target]
y_val = val[target]

with open(inputs["latest_model"], "rb") as f:
    prev = pickle.load(f)

if hasattr(prev, "feature_names_in_"):
    feat_cols = list(prev.feature_names_in_)
else:
    drop = set(targets) | set(excluded)
    if group_col:
        drop.add(group_col)
    feat_cols = [c for c in train.columns if c not in drop]

X_train = train[feat_cols]
X_val = val[feat_cols]

if isinstance(prev, Pipeline):
    prep_steps = prev.steps[:-1]
else:
    prep_steps = []

hgb = HistGradientBoostingClassifier(random_state=42)
new_pipe = Pipeline(steps=prep_steps + [("clf", hgb)])
new_pipe = mlca_runtime.fit_record(new_pipe, X_train, "train", y=y_train)

proba = new_pipe.predict_proba(X_val)
classes = list(new_pipe.classes_)
pos_idx = classes.index(classes[1])
y_bin = (y_val == classes[1]).astype(int)
auc = float(roc_auc_score(y_bin, proba[:, pos_idx]))
mlca_runtime.record_metric("roc_auc", auc, "val")
print("val roc_auc:", round(auc, 4), "| classes:", classes)

mlca_runtime.save_model(new_pipe, "alternative_model.pkl")
