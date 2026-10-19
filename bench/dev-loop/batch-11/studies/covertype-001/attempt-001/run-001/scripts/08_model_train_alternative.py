import mlca_runtime, json, pickle
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
seed = 42
mlca_runtime.set_seed(seed)

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
target = proto["targets"][0]

with open(inp["latest_model"], "rb") as f:
    base = pickle.load(f)

new_est = HistGradientBoostingClassifier(random_state=seed)

# Reuse the most complete saved pipeline's preprocessing; swap in the new estimator.
if isinstance(base, Pipeline) and len(base.steps) > 1:
    pipe = Pipeline(list(base.steps[:-1]) + [("model", new_est)])
else:
    pipe = Pipeline([("model", new_est)])

feats = None
if hasattr(base, "feature_names_in_"):
    feats = list(base.feature_names_in_)
else:
    for _, step in getattr(base, "steps", []):
        if hasattr(step, "feature_names_in_"):
            feats = list(step.feature_names_in_)
            break
if feats is None:
    feats = [c for c in train.columns if c not in proto["targets"] + list(proto["excluded_columns"])]

X = train[feats]
y = train[target]
fitted = mlca_runtime.fit_record(pipe, X, "train", y=y)

mlca_runtime.save_model(fitted, "alternative_model.pkl")

Xv = val[feats]
yv = val[target]
proba = fitted.predict_proba(Xv)
classes = list(fitted.classes_)
scores = []
for i, c in enumerate(classes):
    yt = (yv == c).astype(int)
    if yt.nunique() > 1:
        scores.append(roc_auc_score(yt, proba[:, i]))
auc = sum(scores) / len(scores)
mlca_runtime.record_metric("roc_auc_ovr", float(auc), "val")
print(f"n_train={len(X)} n_feats={len(feats)} n_classes={len(classes)} val_auc={auc:.4f}")
