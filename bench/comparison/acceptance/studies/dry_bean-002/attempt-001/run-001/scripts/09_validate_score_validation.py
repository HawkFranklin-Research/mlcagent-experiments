import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
try:
    task = protocol["task"]
    pm = task["primary_metric"]
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    group = protocol.get("group_column")
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available top-level keys: {list(protocol)}")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
target = targets[0]
y_val = val[target]

drop = set(targets) | set(excluded) | ({group} if group else set())
features = [c for c in train.columns if c not in drop]
for c in list(features):
    s = train[c]
    unique = s.nunique(dropna=True)
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and unique == len(train):
        features.remove(c)
    elif pd.api.types.is_integer_dtype(s) and unique == len(train) and any(t in c.lower() for t in ("id", "code", "no")):
        features.remove(c)

pre = pickle.load(open(inp["05_preprocess_fit_preprocessing_NBMSGCPN/preprocessor.pkl"], "rb"))
sel = pickle.load(open(inp["06_features_select_features_H3ASCQ8N/selector.pkl"], "rb"))

def complete_model(obj):
    if isinstance(obj, Pipeline):
        try:
            obj.predict_proba(val[features].head(2))
            return obj
        except Exception:
            pass
    if isinstance(sel, Pipeline) and sel.named_steps:
        front = [("selector", sel)]
    else:
        front = [("preprocessor", pre), ("selector", sel)]
    return Pipeline(front + [("model", obj)])

def sig(x):
    return float(f"{x:.4g}")

cands = [
    ("logistic", inp["07_model_train_logistic_FG4AYCRT/model.pkl"]),
    ("alternative", inp["08_model_train_alternative_YXJNMVZP/alternative_model.pkl"]),
]
rows = []
best_name, best_score = None, -np.inf
Xv = val[features]

for name, path in cands:
    model = complete_model(pickle.load(open(path, "rb")))
    proba = model.predict_proba(Xv)
    labels = model.classes_ if hasattr(model, "classes_") else model.steps[-1][1].classes_
    auc = roc_auc_score(y_val, proba, labels=labels, multi_class="ovr", average="macro")
    pred = model.predict(Xv)
    f1 = f1_score(y_val, pred, average="macro")
    rows.append((name, pm, sig(auc), np.nan))
    rows.append((name, "f1_macro", sig(f1), np.nan))
    mlca_runtime.record_metric(pm, float(auc), "val")
    if auc > best_score:
        best_name, best_score = name, auc

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics by candidate model")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", sig(best_score), source="validation_metrics.csv")

print(f"rows train={len(train)} val={len(val)} features={len(features)}")
print(f"best_model={best_name} {pm}={best_score:.4f}")
