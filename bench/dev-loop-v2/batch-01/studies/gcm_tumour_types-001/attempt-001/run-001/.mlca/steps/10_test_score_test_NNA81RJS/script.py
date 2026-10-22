import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
task = protocol["task"]
primary = task["primary_metric"]
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", []) or []
group = protocol.get("group_column")
target = targets[0]


def choose_model():
    val = pd.read_csv(inputs["09_validate_score_validation_0B118NSP/validation_metrics.csv"])
    paths = {
        "07": inputs["07_model_train_logistic_FX65AH12/model.pkl"],
        "logistic": inputs["07_model_train_logistic_FX65AH12/model.pkl"],
        "08": inputs["08_model_train_alternative_R7D6ZBXG/alternative_model.pkl"],
        "alternative": inputs["08_model_train_alternative_R7D6ZBXG/alternative_model.pkl"],
    }
    def to_path(x):
        if isinstance(x, str):
            for k, p in paths.items():
                if k in x:
                    return p
        return None
    if "metric" in val.columns:
        sub = val[val["metric"] == primary]
        for col in ["model", "candidate", "estimator", "model_name", "name"]:
            if col in val.columns and len(sub):
                p = to_path(sub.loc[sub["value"].idxmax(), col])
                if p:
                    return p
    for col in ["model", "candidate", "estimator", "model_name", "name"]:
        if col in val.columns and primary in val.columns:
            p = to_path(val.loc[val[primary].idxmax(), col])
            if p:
                return p
    man = json.loads(inputs["09_validate_score_validation_0B118NSP/manifest.json"].read_text())
    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                r = walk(v)
                if r:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = walk(v)
                if r:
                    return r
        elif isinstance(o, str):
            return to_path(o)
        return None
    return walk(man) or inputs["latest_model"]

with open(choose_model(), "rb") as f:
    model = pickle.load(f)

feature_cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
if feature_cols is None and hasattr(model, "steps") and hasattr(model.steps[0][1], "feature_names_in_"):
    feature_cols = list(model.steps[0][1].feature_names_in_)
if feature_cols is None:
    train = pd.read_parquet(inputs["train"])
    drop = set(targets) | set(excluded) | ({group} if group else set())
    feature_cols = []
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        if s.nunique(dropna=True) == len(s) and (
            pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)
        ):
            continue
        feature_cols.append(c)
feature_cols = [c for c in feature_cols if c not in set(targets) | set(excluded) | ({group} if group else set())]

test_df = pd.read_parquet(inputs["test"])
X_test = test_df[feature_cols]
y_true = test_df[target].values
proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)
clf = model.steps[-1][1] if hasattr(model, "steps") else model
classes = np.asarray(clf.classes_)


def macro_ovr_auc(y, p, cls):
    present = [c for c in cls if np.any(y == c)]
    if len(present) == len(cls) and len(np.unique(y)) > 1:
        try:
            return roc_auc_score(y, p, multi_class="ovr", average="macro", labels=cls)
        except Exception:
            pass
    vals = []
    for j, c in enumerate(cls):
        yb = (y == c).astype(int)
        if yb.min() == yb.max():
            continue
        try:
            vals.append(roc_auc_score(yb, p[:, j]))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else np.nan


def macro_f1(y, pred):
    return f1_score(y, pred, average="macro", zero_division=0)


metric_names = [primary] if primary == "macro_f1" else [primary, "macro_f1"]
values = {}
for name in metric_names:
    values[name] = macro_ovr_auc(y_true, proba, classes) if name == "roc_auc_ovr" else macro_f1(y_true, y_pred)

rng = np.random.default_rng(42)
boot = {name: [] for name in metric_names}
for _ in range(1000):
    idx = rng.integers(0, len(y_true), len(y_true))
    yb, pb, predb = y_true[idx], proba[idx], y_pred[idx]
    for name in metric_names:
        boot[name].append(macro_ovr_auc(yb, pb, classes) if name == "roc_auc_ovr" else macro_f1(yb, predb))

ci = {}
for name in metric_names:
    arr = np.asarray(boot[name], dtype=float)
    arr = arr[~np.isnan(arr)]
    ci[name] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if len(arr) else (np.nan, np.nan)

rows = [{"metric": name, "value": values[name], "ci_low": ci[name][0], "ci_high": ci[name][1]} for name in metric_names]
metrics_df = pd.DataFrame(rows).round(4)
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")
mlca_runtime.record_metric(primary, float(values[primary]), split="test", ci=list(ci[primary]))
if primary == "roc_auc":
    mlca_runtime.record_fact_candidate("test_roc_auc", float(values[primary]), source="test_metrics.csv")

pred_df = pd.DataFrame(index=test_df.index)
pred_df["row_id"] = test_df.index
pred_df["partition"] = "test"
for t in targets:
    pred_df[f"y_true__{t}"] = test_df[t].values
    pred_df[f"y_pred__{t}"] = y_pred
for j, c in enumerate(classes):
    pred_df[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred_df, "test_predictions.parquet")

print("primary_metric", primary)
print("test_rows", len(test_df))
print("classes", len(classes))
print("headline", values[primary], ci[primary])
