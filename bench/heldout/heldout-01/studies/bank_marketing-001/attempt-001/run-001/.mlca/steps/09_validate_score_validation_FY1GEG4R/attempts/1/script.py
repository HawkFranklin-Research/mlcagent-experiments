import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             f1_score, roc_curve)
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
for k in ("targets", "task"):
    if k not in protocol:
        raise KeyError(f"Missing protocol key {k}; available: {list(protocol.keys())}")
targets = protocol["targets"]
excluded = list(protocol.get("excluded_columns") or [])
group_col = protocol.get("group_column")
task = protocol["task"]
primary_metric = task["primary_metric"]
task_type = task["type"]

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

feature_cols = [c for c in train.columns if c not in targets
                and c not in excluded and c != group_col]
drop = []
for c in feature_cols:
    s = train[c]
    if s.nunique(dropna=False) == len(s):
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s.dtype):
            drop.append(c)
        elif pd.api.types.is_integer_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype):
            drop.append(c)
feature_cols = [c for c in feature_cols if c not in drop]

pre = pickle.load(open(paths["05_preprocess_fit_preprocessing_6KJVP4ST/preprocessor.pkl"], "rb"))
sel = pickle.load(open(paths["06_features_select_features_TX9DR60B/selector.pkl"], "rb"))
raw_models = {
    "logistic": pickle.load(open(paths["07_model_train_logistic_GD4X7DTC/model.pkl"], "rb")),
    "alternative": pickle.load(open(paths["08_model_train_alternative_0A2EAXQK/alternative_model.pkl"], "rb")),
}

def build_pipeline(m):
    if isinstance(m, Pipeline) and hasattr(m, "feature_names_in_"):
        return m
    pre_f = getattr(pre, "feature_names_in_", None)
    sel_f = getattr(sel, "feature_names_in_", None)
    steps = []
    if pre_f is not None and sel_f is not None and set(pre_f) == set(sel_f):
        steps.append(("frontend", sel))
    else:
        if pre is not None:
            steps.append(("pre", pre))
        if sel is not None:
            steps.append(("sel", sel))
    steps.append(("model", m))
    return Pipeline(steps)

models = {k: build_pipeline(v) for k, v in raw_models.items()}
y_val = val[targets[0]]
rows = []

for name, m in models.items():
    feats = getattr(m, "feature_names_in_", None)
    if feats is None:
        first = list(m.named_steps.values())[0]
        feats = getattr(first, "feature_names_in_", feature_cols)
    feats = list(feats)
    X = val[feats]
    proba = m.predict_proba(X)
    classes = list(m.classes_)
    if task_type == "binary":
        pos = protocol.get("positive_label")
        if pos is None:
            pos = classes[1]
        j = classes.index(pos)
        y_bin = (y_val == pos).astype(int).values
        p = proba[:, j]
        auc = roc_auc_score(y_bin, p)
        ap = average_precision_score(y_bin, p)
        brier = brier_score_loss(y_bin, p)
        fpr, tpr, thr = roc_curve(y_bin, p)
        best_thr = float(thr[int(np.argmax(tpr - fpr))])
        bal = balanced_accuracy_score(y_bin, (p >= best_thr).astype(int))
        scores = {"roc_auc": auc, "pr_auc": ap,
                  "balanced_accuracy": bal, "brier_score": brier}
        if primary_metric not in scores:
            scores[primary_metric] = auc
        for mname, v in scores.items():
            if mname in ("roc_auc", "balanced_accuracy", primary_metric):
                tv = round(best_thr, 4)
            else:
                tv = np.nan
            rows.append({"model": name, "metric": mname,
                         "value": round(float(v), 4), "threshold": tv})
    else:
        y_arr = y_val.values
        auc_m = float("nan")
        try:
            auc_m = roc_auc_score(y_arr, proba, multi_class="ovr",
                                  average="macro", labels=classes)
        except ValueError:
            present = [i for i, c in enumerate(classes) if (y_arr == c).any()]
            vals = []
            for i in present:
                yi = (y_arr == classes[i]).astype(int)
                if 0 < yi.sum() < len(yi):
                    vals.append(roc_auc_score(yi, proba[:, i]))
            auc_m = float(np.mean(vals)) if vals else float("nan")
            print(f"{name} classes excluded from OVR: {[c for i,c in enumerate(classes) if i not in present]}")
        y_pred = np.array(classes)[proba.argmax(axis=1)]
        f1m = f1_score(y_arr, y_pred, average="macro")
        scores = {"roc_auc_ovr": auc_m, "f1_macro": f1m}
        if primary_metric not in scores:
            scores[primary_metric] = auc_m
        for mname, v in scores.items():
            rows.append({"model": name, "metric": mname,
                         "value": round(float(v), 4), "threshold": np.nan})

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "validation_metrics.csv", "Validation metrics per model")

pm = df[df["metric"] == primary_metric]
higher_better = primary_metric in ("roc_auc", "pr_auc", "roc_auc_ovr",
                                   "balanced_accuracy", "f1_macro",
                                   "accuracy", "f1")
best_idx = pm["value"].idxmax() if higher_better else pm["value"].idxmin()
best_model = str(pm.loc[best_idx, "model"])
best_val = float(pm.loc[best_idx, "value"])

auc_key = "roc_auc" if task_type == "binary" else "roc_auc_ovr"
arows = df[(df["metric"] == auc_key) & (df["model"] == best_model)]
best_auc = float(arows["value"].iloc[0])

mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")
mlca_runtime.record_metric(f"val_{primary_metric}", best_val, "val")
print(f"models={len(models)} rows={len(df)} best={best_model} {auc_key}={best_auc:.4f}")
