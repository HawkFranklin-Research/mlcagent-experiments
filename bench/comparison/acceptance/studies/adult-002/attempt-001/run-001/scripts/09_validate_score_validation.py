import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, roc_curve, f1_score)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
try:
    task = protocol["task"]
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    primary = task["primary_metric"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available top-level keys: {sorted(protocol)}")
group_col = protocol.get("group_column")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)

feat_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or
            isinstance(s.dtype, pd.CategoricalDtype)) and s.nunique(dropna=True) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s) and (
            c.lower().endswith("id") or c.lower().endswith("key")):
        continue
    feat_cols.append(c)

def load(name):
    with open(inp[name], "rb") as f:
        return pickle.load(f)

pre = load("05_preprocess_fit_preprocessing_5C9G845H/preprocessor.pkl")
sel = load("06_features_select_features_HP31G2ZF/selector.pkl")
models = {
    "logistic_QH9VENHA": load("07_model_train_logistic_QH9VENHA/model.pkl"),
    "alternative_2548TA6F": load("08_model_train_alternative_2548TA6F/alternative_model.pkl"),
}

def first_cols(obj):
    if isinstance(obj, Pipeline):
        for _, st in obj.steps:
            cols = getattr(st, "feature_names_in_", None)
            if cols is not None:
                return list(cols)
            if isinstance(st, Pipeline):
                c = first_cols(st)
                if c:
                    return c
        cols = getattr(obj, "feature_names_in_", None)
        return list(cols) if cols is not None else None
    cols = getattr(obj, "feature_names_in_", None)
    return list(cols) if cols is not None else None

def predictor_for(model_obj):
    if isinstance(model_obj, Pipeline) and hasattr(model_obj, "predict_proba"):
        c = first_cols(model_obj)
        if c is None or set(c) == set(feat_cols):
            return model_obj, c or feat_cols
        if set(c).issubset(set(feat_cols)):
            return model_obj, c
    scols = first_cols(sel)
    if isinstance(sel, Pipeline) and scols is not None and set(scols) == set(feat_cols):
        front = sel
    else:
        pcols = first_cols(pre)
        if pcols is not None and set(pcols) == set(feat_cols):
            if sel is None or (isinstance(sel, Pipeline) and scols is not None and set(scols) == set(feat_cols)):
                front = sel if isinstance(sel, Pipeline) and scols is not None and set(scols) == set(feat_cols) else pre
            else:
                front = Pipeline([("preprocessor", pre), ("selector", sel)])
        else:
            front = pre
    steps = (front.steps if isinstance(front, Pipeline) else [("frontend", front)])
    steps = steps + (model_obj.steps if isinstance(model_obj, Pipeline) else [("model", model_obj)])
    return Pipeline(steps), feat_cols

def classes_of(p):
    if hasattr(p, "classes_"):
        return np.asarray(p.classes_)
    return np.asarray(p.steps[-1][1].classes_)

def metrics_for(name, model_obj):
    pred, cols = predictor_for(model_obj)
    X = val[cols]
    proba = pred.predict_proba(X)
    classes = classes_of(pred)
    y = val[targets[0]].to_numpy()
    rows = []
    if len(classes) == 2:
        pos = task.get("positive_label", protocol.get("positive_label"))
        if pos is None:
            pos = classes[1]
        pos = np.asarray(classes)[list(classes).index(pos)]
        y_bin = (y == pos).astype(int)
        p = proba[:, list(classes).index(pos)]
        auc = roc_auc_score(y_bin, p)
        pr = average_precision_score(y_bin, p)
        fpr, tpr, thr = roc_curve(y_bin, p)
        t = float(thr[int(np.argmax(tpr - fpr))])
        if not np.isfinite(t):
            t = float(np.max(p))
        bal = balanced_accuracy_score(y_bin, (p >= t).astype(int))
        brier = brier_score_loss(y_bin, p)
        rows = [("roc_auc", auc, np.nan), ("pr_auc", pr, np.nan),
                ("balanced_accuracy", bal, t), ("brier_score", brier, np.nan)]
    else:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)
        pred_lab = pred.predict(X)
        f1 = f1_score(y, pred_lab, average="macro", labels=classes)
        rows = [("roc_auc_ovr", auc, np.nan), ("f1_macro", f1, np.nan)]
    return [(name, m, v, t) for m, v, t in rows]

rows = []
for name, model_obj in models.items():
    rows.extend(metrics_for(name, model_obj))

out = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
def sig4(x):
    return np.nan if pd.isna(x) else float(f"{x:.4g}")
out["value"] = out["value"].map(sig4)
out["threshold"] = out["threshold"].map(sig4)
mlca_runtime.save_table(out, "validation_metrics.csv", caption="Validation metrics")

lower_better = any(k in primary.lower() for k in ["brier", "log_loss", "mae", "mse", "rmse"])
prows = out[out["metric"] == primary]
if prows.empty:
    prows = out[out["metric"] == ("roc_auc_ovr" if "roc_auc_ovr" in set(out["metric"]) else primary)]
best_row = prows.sort_values("value", ascending=not lower_better).iloc[-1] if lower_better else prows.sort_values("value", ascending=False).iloc[0]
for _, r in prows.sort_values("value", ascending=not lower_better).iterrows():
    mlca_runtime.record_metric(primary, float(r["value"]), split="val")
best_model = best_row["model"]
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
auc_row = out[(out["model"] == best_model) & (out["metric"] == "roc_auc")]
if not auc_row.empty:
    mlca_runtime.record_fact_candidate("best_val_roc_auc", float(auc_row.iloc[0]["value"]), source="validation_metrics.csv")
print(f"train={train.shape} val={val.shape} features={len(feat_cols)}")
print(f"best_model={best_model} {primary}={best_row['value']:.4g}")
