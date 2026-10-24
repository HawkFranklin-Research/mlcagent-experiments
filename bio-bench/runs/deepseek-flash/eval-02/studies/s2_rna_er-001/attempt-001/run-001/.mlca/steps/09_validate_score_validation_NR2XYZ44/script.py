import mlca_runtime
import json, pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
    balanced_accuracy_score, brier_score_loss, f1_score, roc_curve)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
task = proto["task"]
primary = task["primary_metric"]
targets = list(proto["targets"])
excluded = set(proto["excluded_columns"])
group_col = proto.get("group_column")
target_col = targets[0]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

feat_cols = [c for c in train.columns if c not in set(targets) | excluded | ({group_col} if group_col else set())]

def load_pkl(k):
    with open(inp[k], "rb") as f:
        return pickle.load(f)

preproc = load_pkl("05_preprocess_fit_preprocessing_R5N312M1/preprocessor.pkl")
sel = load_pkl("06_features_select_features_JQQFHDQ8/selector.pkl")
models = {
    "logistic_regression": load_pkl("07_model_train_logistic_8JHDFT9S/model.pkl"),
    "alternative": load_pkl("08_model_train_alternative_55567D2N/alternative_model.pkl"),
}

def first_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        first = obj.named_steps[list(obj.named_steps)[0]]
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
    return None

def as_df(a, idx):
    return a if isinstance(a, pd.DataFrame) else pd.DataFrame(np.asarray(a), index=idx)

def to_input(obj, X):
    names = first_names(obj)
    return X[names].copy() if names is not None else X[feat_cols].copy()

def proba_of(model, X):
    if isinstance(model, Pipeline):
        return model.predict_proba(to_input(model, X))
    if isinstance(sel, Pipeline):
        Xt = as_df(sel.transform(to_input(sel, X)), X.index)
    else:
        Xp = as_df(preproc.transform(to_input(preproc, X)), X.index)
        Xt = as_df(sel.transform(to_input(sel, Xp)), X.index)
    return model.predict_proba(Xt)

def classes_of(model):
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    if isinstance(model, Pipeline):
        last = model.named_steps[list(model.named_steps)[-1]]
        return np.asarray(last.classes_)
    raise RuntimeError("classes_ unavailable")

y_val = val[target_col].to_numpy()
metric_rows = {}
for name, model in models.items():
    P = proba_of(model, val)
    cls = classes_of(model)
    if len(cls) == 2:
        pos = task.get("positive_label", cls[1])
        j = int(np.where(cls == pos)[0][0])
        p = P[:, j]
        yb = (y_val == pos).astype(int)
        roc = float(roc_auc_score(yb, p))
        pr = float(average_precision_score(yb, p))
        br = float(brier_score_loss(yb, p))
        fpr, tpr, thr = roc_curve(yb, p)
        bthr = float(thr[np.argmax(tpr - fpr)])
        ba = float(balanced_accuracy_score(yb, (p >= bthr).astype(int)))
        metric_rows[name] = {"roc_auc": (roc, np.nan), "pr_auc": (pr, np.nan),
                             "balanced_accuracy": (ba, bthr), "brier_score": (br, np.nan)}
    else:
        yi = pd.Categorical(y_val, categories=list(cls)).codes.astype(int)
        roc = float(roc_auc_score(yi, P, multi_class="ovr", average="macro",
                                  labels=list(range(len(cls)))))
        yp = cls[P.argmax(axis=1)]
        f1 = float(f1_score(y_val, yp, average="macro"))
        metric_rows[name] = {"roc_auc": (roc, np.nan), "f1_macro": (f1, np.nan)}

tbl = [{"model": m, "metric": k, "value": v[0], "threshold": v[1]}
       for m, d in metric_rows.items() for k, v in d.items()]
df = pd.DataFrame(tbl)
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics by candidate model.")

for m in models:
    v = df[(df.model == m) & (df.metric == primary)]["value"]
    if len(v):
        mlca_runtime.record_metric(primary, float(v.iloc[0]), split="val")

higher_better = primary in ("roc_auc", "pr_auc", "balanced_accuracy", "f1_macro",
                            "accuracy", "f1", "recall", "precision")
sub = df[df.metric == primary].sort_values("value", ascending=not higher_better)
best = sub.iloc[0]["model"]
val_auc = float(df[(df.model == best) & (df.metric == "roc_auc")]["value"].iloc[0])

mlca_runtime.record_fact_candidate("best_model", best, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", val_auc, source="validation_metrics.csv")

print(f"train={len(train)} val={len(val)} best={best} val_roc_auc={val_auc:.4f}")
