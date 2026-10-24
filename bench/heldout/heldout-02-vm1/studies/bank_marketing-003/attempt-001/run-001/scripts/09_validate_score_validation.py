import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, roc_curve

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in proto:
        raise KeyError(f"Missing protocol key {key!r}; available: {list(proto)}")
targets = proto["targets"]
excluded = proto["excluded_columns"]
group = proto["group_column"]
task = proto["task"]
primary = task["primary_metric"]
if task.get("type", "binary") != "binary":
    raise ValueError("This validation script handles binary tasks")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
raw_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
for c in list(raw_cols):
    s = train[c]
    if s.notna().all() and s.nunique(dropna=True) == len(train):
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype)
                or (pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s))):
            raw_cols.remove(c)
X_val = val[raw_cols]
target_col = targets[0]
y_val = val[target_col].copy()

def load_pkl(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("05_preprocess_fit_preprocessing_CKA3B6GT/preprocessor.pkl")
sel = load_pkl("06_features_select_features_XKN9FXQ4/selector.pkl")
models = {
    "logistic": load_pkl("07_model_train_logistic_5PDZPGWS/model.pkl"),
    "alternative": load_pkl("latest_model"),
}

class Chain:
    def __init__(self, steps):
        self.steps = steps
        first = steps[0][1]
        if hasattr(first, "feature_names_in_"):
            self.feature_names_in_ = first.feature_names_in_
        if hasattr(first, "n_features_in_"):
            self.n_features_in_ = first.n_features_in_
    def predict_proba(self, X):
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        return self.steps[-1][1].predict_proba(Xt)
    @property
    def classes_(self):
        return self.steps[-1][1].classes_

def raw_ready(obj):
    if obj is None:
        return False
    if hasattr(obj, "feature_names_in_"):
        exp = list(obj.feature_names_in_)
        return len(exp) > 0 and all(c in raw_cols for c in exp)
    if hasattr(obj, "n_features_in_") and hasattr(obj, "named_steps"):
        return obj.n_features_in_ == len(raw_cols)
    return False

def make_candidate(model):
    opts = []
    if raw_ready(model):
        opts.append(model)
    if sel is not None and hasattr(sel, "transform"):
        opts.append(Chain([("selector", sel), ("model", model)]))
    if pre is not None and hasattr(pre, "transform"):
        opts.append(Chain([("preprocessor", pre), ("model", model)]))
    if pre is not None and sel is not None and hasattr(pre, "transform") and hasattr(sel, "transform"):
        opts.append(Chain([("preprocessor", pre), ("selector", sel), ("model", model)]))
    probe = X_val.iloc[: min(2, len(X_val))]
    for opt in opts:
        if raw_ready(opt):
            try:
                opt.predict_proba(probe)
                return opt
            except Exception:
                pass
    for opt in opts:
        try:
            opt.predict_proba(probe)
            return opt
        except Exception:
            pass
    raise RuntimeError("No compatible saved preprocessing/selector/model pipeline")

def proba_from(cand, X):
    if hasattr(cand, "feature_names_in_"):
        X = X[list(cand.feature_names_in_)]
    return cand.predict_proba(X)

rows, results = [], {}
for name, model in models.items():
    cand = make_candidate(model)
    proba = proba_from(cand, X_val)
    classes = list(cand.classes_)
    pos = task.get("positive_label") or proto.get("positive_label") or classes[1]
    if pos not in classes:
        pos = classes[1]
    p = proba[:, classes.index(pos)]
    yb = (y_val == pos).astype(int).to_numpy()
    fpr, tpr, thr = roc_curve(yb, p)
    j = tpr - fpr
    finite = np.isfinite(thr)
    best_thr = float(thr[finite][np.argmax(j[finite])]) if finite.any() else 0.5
    vals = {
        "roc_auc": roc_auc_score(yb, p),
        "pr_auc": average_precision_score(yb, p),
        "balanced_accuracy": balanced_accuracy_score(yb, (p >= best_thr).astype(int)),
        "brier_score": brier_score_loss(yb, p),
    }
    results[name] = {"metrics": vals, "threshold": best_thr}
    for metric in ["roc_auc", "pr_auc", "balanced_accuracy", "brier_score"]:
        rows.append({
            "model": name,
            "metric": metric,
            "value": vals[metric],
            "threshold": best_thr if metric == "balanced_accuracy" else np.nan,
        })

metrics_df = pd.DataFrame(rows)
for c in ["value", "threshold"]:
    metrics_df[c] = metrics_df[c].map(lambda x: np.nan if pd.isna(x) else float(f"{x:.4g}"))
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics by candidate model")

higher_better = primary not in {"brier_score", "log_loss", "mae", "rmse", "mse"}
best_name = (max if higher_better else min)(results, key=lambda k: results[k]["metrics"][primary])
best_val = results[best_name]["metrics"][primary]

for name in results:
    mlca_runtime.record_metric(primary, results[name]["metrics"][primary], "val")
mlca_runtime.record_metric(primary, best_val, "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
if primary == "roc_auc":
    mlca_runtime.record_fact_candidate("best_val_roc_auc", results[best_name]["metrics"]["roc_auc"], source="validation_metrics.csv")

print(f"val rows={len(val)}, features={len(raw_cols)}")
print(f"best={best_name} {primary}={best_val:.4f}")
