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
targets = protocol["targets"]
excluded = list(protocol.get("excluded_columns", []))
group_col = protocol.get("group_column")
task = protocol["task"]
metric = task["primary_metric"]
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

pre = pickle.loads(inputs["05_preprocess_fit_preprocessing_M7A7QE51/preprocessor.pkl"].read_bytes())
sel = pickle.loads(inputs["06_features_select_features_8STHQFS1/selector.pkl"].read_bytes())

def has_ct(obj):
    if isinstance(obj, ColumnTransformer):
        return True
    if isinstance(obj, Pipeline):
        return any(has_ct(s) for _, s in obj.named_steps.items())
    return False

clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
if has_ct(sel):
    pipeline = Pipeline([("sel", sel), ("clf", clf)])
else:
    pipeline = Pipeline([("prep", pre), ("sel", sel), ("clf", clf)])

if hasattr(pre, "feature_names_in_"):
    feature_cols = list(pre.feature_names_in_)
else:
    feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
feature_cols = [c for c in feature_cols if c not in targets and c not in excluded and c != group_col]

X_train = train[feature_cols]
y_train = train[target]
X_val = val[feature_cols]
y_val = val[target]

pipeline = mlca_runtime.fit_record(pipeline, X_train, "train", y=y_train)

proba = pipeline.predict_proba(X_val)
classes = pipeline.named_steps["clf"].classes_
present = [c for c in classes if c in set(y_val.tolist())]
if len(present) == len(classes):
    auc = roc_auc_score(y_val, proba, labels=classes, multi_class="ovr", average="macro")
else:
    per_class = [roc_auc_score((y_val == c).astype(int), proba[:, j])
                 for j, c in enumerate(classes) if c in present]
    auc = float(np.mean(per_class))

mlca_runtime.record_metric(metric, float(auc), "val")
mlca_runtime.save_model(pipeline, "model.pkl")
print(f"train_rows={len(X_train)} features={len(feature_cols)} classes={len(classes)}")
print(f"val_{metric}={auc:.4f}")
