import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss, roc_curve)

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
task = protocol["task"]
targets = protocol["targets"] if isinstance(protocol["targets"], list) else [protocol["targets"]]
excluded = list(protocol.get("excluded_columns") or [])
group_col = protocol.get("group_column")
primary = task["primary_metric"]
is_binary = task.get("type", "binary") == "binary"

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
pre = load_pkl(paths["05_preprocess_fit_preprocessing_XT8NJ07Q/preprocessor.pkl"])
sel = load_pkl(paths["06_features_select_features_A3NFBXN9/selector.pkl"])

def feature_names_of(obj):
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        try:
            return list(obj.feature_names_in_)
        except Exception:
            pass
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            n = feature_names_of(step)
            if n is not None:
                return n
    return None

feat_cols = feature_names_of(pre)
if feat_cols is None:
    feat_cols = [c for c in train.columns
                 if c not in targets and c not in excluded and c != group_col]

def assemble(model):
    if isinstance(model, Pipeline):
        names = [n.lower() for n in model.named_steps.keys()]
        has_pre = any("pre" in n or "transform" in n for n in names)
        has_sel = sel is not None and any("select" in n for n in names)
        if has_pre and (has_sel or sel is None):
            return model
        pre_steps = [("pre_o", pre)]
        if not has_sel and sel is not None:
            pre_steps.append(("sel_o", sel))
        if has_pre:
            return Pipeline(pre_steps + list(model.steps))
        return Pipeline(pre_steps + list(model.steps))
    steps = [("pre_o", pre)]
    if sel is not None:
        steps.append(("sel_o", sel))
    steps.append(("model", model))
    return Pipeline(steps)

model_paths = {
    "logistic": paths["07_model_train_logistic_QY4T2JSR/model.pkl"],
    "alternative": paths["08_model_train_alternative_5C3RWNJ3/alternative_model.pkl"],
}
models = {n: (load_pkl(p), None) for n, p in model_paths.items()}
models = {n: (m, assemble(m)) for n, (m, _) in models.items()}

X_val = val[feat_cols].copy()
y_val = val[targets[0]]

pos_label = task.get("positive_label") or task.get("positive_class")
if pos_label is None:
    for _, (raw, full) in models.items():
        est = full.steps[-1][1]
        classes = getattr(est, "classes_", None)
        if classes is not None:
            pos_label = classes[1]
            break

rows, aucs = [], {}
for name, (raw, full) in models.items():
    est = full.steps[-1][1]
    classes = np.asarray(getattr(est, "classes_", None))
    proba = full.predict_proba(X_val)
    col = int(np.where(classes == pos_label)[0][0]) if classes is not None and pos_label in classes else 1
    p = proba[:, col]
    y_bin = (y_val == pos_label).astype(int).to_numpy()

    if is_binary:
        auc = float(roc_auc_score(y_bin, p))
        pr = float(average_precision_score(y_bin, p))
        fpr, tpr, thr = roc_curve(y_bin, p)
        youden = float(thr[int(np.argmax(tpr - fpr))])
        ba = float(balanced_accuracy_score(y_bin, (p >= youden).astype(int)))
        brier = float(brier_score_loss(y_bin, p))
        rows += [
            {"model": name, "metric": "roc_auc", "value": auc, "threshold": np.nan},
            {"model": name, "metric": "pr_auc", "value": pr, "threshold": np.nan},
            {"model": name, "metric": "balanced_accuracy", "value": ba, "threshold": youden},
            {"model": name, "metric": "brier", "value": brier, "threshold": np.nan},
        ]
        aucs[name] = auc

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics per candidate model.")

best_model = max(aucs, key=aucs.get) if aucs else None
if best_model is not None:
    mlca_runtime.record_metric(primary, aucs[best_model], "val")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
if best_model is not None:
    mlca_runtime.record_fact_candidate("best_val_roc_auc", aucs[best_model],
                                       source="validation_metrics.csv")

print(f"val rows: {len(X_val)}, models: {len(models)}")
print(f"best: {best_model}, {primary}={aucs.get(best_model)}")
