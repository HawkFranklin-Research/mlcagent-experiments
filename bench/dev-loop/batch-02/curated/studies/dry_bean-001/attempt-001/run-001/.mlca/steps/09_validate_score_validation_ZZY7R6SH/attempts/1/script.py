import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             roc_curve)
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

proto = json.loads(inputs["_protocol"].read_text())
targets = proto["targets"]
excluded = proto.get("excluded_columns", [])
group = proto.get("group_column")
task = proto["task"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
target = targets[0]

feat_cols = [c for c in train.columns
             if c not in targets and c not in excluded and c != group]
drop = []
for c in feat_cols:
    s = train[c]
    if s.dtype == object and s.nunique() == len(s):
        drop.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique() == len(s):
        drop.append(c)
feat_cols = [c for c in feat_cols if c not in drop]

X_val = val[feat_cols]
y_val = val[target]

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load(inputs["05_preprocess_fit_preprocessing_CQ5BZKTY/preprocessor.pkl"])
sel = load(inputs["06_features_select_features_XJNGW7H2/selector.pkl"])
models = {
    "logistic": load(inputs["07_model_train_logistic_8D243SEH/model.pkl"]),
    "alternative": load(inputs["08_model_train_alternative_052PRAHM/alternative_model.pkl"]),
}

def as_pipeline(m):
    if isinstance(m, Pipeline):
        return m
    return Pipeline([("pre", pre), ("sel", sel), ("clf", m)])

rows = []
for name, m in models.items():
    pipe = as_pipeline(m)
    proba = pipe.predict_proba(X_val)
    classes = list(pipe.classes_)
    pred = pipe.predict(X_val)
    n_classes = len(classes)
    if n_classes == 2:
        pos = classes[1]
        p = proba[:, classes.index(pos)]
        roc = roc_auc_score(y_val, p)
        pr = average_precision_score(y_val, p)
        ba = balanced_accuracy_score(y_val, pred)
        brier = brier_score_loss(y_val, p)
        fpr, tpr, thr = roc_curve(y_val, p)
        k = int(np.argmax(tpr - fpr))
        t = float(thr[k])
        yhat = (p >= t).astype(int)
        ypos = (y_val == pos).astype(int)
        tp = int(((yhat == 1) & (ypos == 1)).sum())
        fn = int(((yhat == 0) & (ypos == 1)).sum())
        tn = int(((yhat == 0) & (ypos == 0)).sum())
        fp = int(((yhat == 1) & (ypos == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
    else:
        roc = roc_auc_score(y_val, proba, multi_class="ovr",
                            average="macro", labels=classes)
        Y = pd.get_dummies(y_val).reindex(columns=classes, fill_value=0).values
        pr = average_precision_score(Y, proba, average="macro")
        ba = balanced_accuracy_score(y_val, pred)
        brier = float(np.mean(np.sum((proba - Y) ** 2, axis=1)))
        t = np.nan
        sens = np.nan
        spec = np.nan
    for metric, value in [("roc_auc", roc), ("pr_auc", pr),
                          ("balanced_accuracy", ba), ("brier", brier),
                          ("sensitivity", sens), ("specificity", spec)]:
        rows.append({"model": name, "metric": metric,
                     "value": round(float(value), 6) if value == value else np.nan,
                     "threshold": t})

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "validation_metrics.csv", "Validation metrics per model")

for name in models:
    v = df[(df.model == name) & (df.metric == "roc_auc")]["value"].iloc[0]
    mlca_runtime.record_metric(f"roc_auc_{name}", float(v), "val")

best = df[df.metric == "roc_auc"].sort_values("value", ascending=False).iloc[0]
mlca_runtime.record_fact_candidate("best_model", best["model"], source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best["value"]), source="validation_metrics.csv")

print(f"features={len(feat_cols)} models={len(models)}")
print(f"best={best['model']} roc_auc={best['value']:.4f}")
