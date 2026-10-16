import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss, roc_curve)
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

proto = json.loads(inputs["_protocol"].read_text())
targets = proto["targets"]
excluded = proto.get("excluded_columns", []) or []
group = proto.get("group_column")
task = proto["task"]
pos_label = task.get("positive_label")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
target = targets[0]
y_val = val[target]

feat_cols = [c for c in train.columns
             if c not in targets and c not in excluded and c != group]
drop = []
for c in feat_cols:
    s = train[c]
    if s.dtype == object:
        drop.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique() == len(s):
        drop.append(c)
feat_cols = [c for c in feat_cols if c not in drop]
X_val = val[feat_cols]

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load(inputs["05_preprocess_fit_preprocessing_BKKYG4P6/preprocessor.pkl"])
sel = load(inputs["06_features_select_features_GQKA7M4V/selector.pkl"])
m1 = load(inputs["07_model_train_logistic_7WNJSTAG/model.pkl"])
m2 = load(inputs["08_model_train_alternative_84VF3NXG/alternative_model.pkl"])

def make_pipe(model):
    if isinstance(model, Pipeline):
        return model
    front = sel if isinstance(sel, Pipeline) else Pipeline([("pre", pre), ("sel", sel)])
    return Pipeline([("front", front), ("clf", model)])

models = {"logistic": m1, "alternative": m2}
rows = []
for name, m in models.items():
    pipe = make_pipe(m)
    proba = pipe.predict_proba(X_val)
    classes = list(pipe.classes_)
    pl = pos_label if pos_label is not None else classes[1]
    p = proba[:, classes.index(pl)]
    yb = (y_val == pl).astype(int)
    roc = roc_auc_score(yb, p)
    pr = average_precision_score(yb, p)
    fpr, tpr, thr = roc_curve(yb, p)
    k = int(np.argmax(tpr - fpr))
    th = float(thr[k])
    pred = (p >= th).astype(int)
    ba = balanced_accuracy_score(yb, pred)
    brier = brier_score_loss(yb, p)
    sens, spec = float(tpr[k]), float(1 - fpr[k])
    for metric, v in [("roc_auc", roc), ("pr_auc", pr), ("balanced_accuracy", ba),
                      ("brier", brier), ("sensitivity", sens), ("specificity", spec)]:
        rows.append({"model": name, "metric": metric, "value": float(v), "threshold": th})
    mlca_runtime.record_metric(f"roc_auc_{name}", float(roc), "val")

df = pd.DataFrame(rows)
df["value"] = df["value"].round(4)
df["threshold"] = df["threshold"].round(4)
mlca_runtime.save_table(df, "validation_metrics.csv", "Validation metrics per model")

best = df[df.metric == "roc_auc"].sort_values("value", ascending=False).iloc[0]
mlca_runtime.record_fact_candidate("best_model", best["model"], source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best["value"]), source="validation_metrics.csv")

print("val rows:", len(val), "features:", len(feat_cols))
print("best:", best["model"], "roc_auc:", float(best["value"]))
