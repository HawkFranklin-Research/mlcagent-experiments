import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

for key in ("task", "targets", "excluded_columns"):
    if key not in protocol:
        raise KeyError(f"Missing {key}; available top-level keys: {list(protocol.keys())}")

task = protocol["task"]
primary = task["primary_metric"]
targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
test = pd.read_parquet(paths["test"])

def is_keylike(s):
    if s.dtype == object or str(s.dtype) in ("string", "category"):
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == len(s)
    return False

feature_cols = [
    c for c in train.columns
    if c not in set(targets) | excluded | ({group} if group else set())
    and not is_keylike(train[c])
]
target = targets[0]

with open(paths["05_preprocess_fit_preprocessing_NBMSGCPN/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(paths["06_features_select_features_H3ASCQ8N/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def frontend():
    if isinstance(sel, Pipeline) and len(sel.named_steps) > 1:
        return sel
    return Pipeline([("pre", pre), ("sel", sel)])

def make_full(mdl):
    if isinstance(mdl, Pipeline) and len(mdl.named_steps) > 1:
        return mdl
    return Pipeline([("front", frontend()), ("model", mdl)])

def get_classes(m):
    if isinstance(m, Pipeline):
        last = m.named_steps[list(m.named_steps)[-1]]
        return last.classes_
    return m.classes_

cand_keys = [
    "07_model_train_logistic_FG4AYCRT/model.pkl",
    "08_model_train_alternative_YXJNMVZP/alternative_model.pkl",
]
models = {}
for key in cand_keys:
    with open(paths[key], "rb") as f:
        models[key] = make_full(pickle.load(f))

Xv = val[feature_cols]
yv = val[target]
best_key, best_val = None, -np.inf
for key, model in models.items():
    p = model.predict_proba(Xv)
    s = roc_auc_score(yv, p, multi_class="ovr", average="macro", labels=get_classes(model))
    if s > best_val:
        best_key, best_val = key, s

model = models[best_key]
Xte = test[feature_cols]
yte = test[target]
proba = model.predict_proba(Xte)
classes = get_classes(model)
pred = classes[np.argmax(proba, axis=1)]

auc = roc_auc_score(yte, proba, multi_class="ovr", average="macro", labels=classes)
f1 = f1_score(yte, pred, average="macro", labels=classes, zero_division=0)

rng = np.random.default_rng(42)
yte_arr = yte.to_numpy()
idx_by_class = [np.flatnonzero(yte_arr == c) for c in classes]
b_auc, b_f1 = [], []
for _ in range(1000):
    idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx_by_class])
    yb = yte_arr[idx]
    pb = proba[idx, :]
    pb_pred = classes[np.argmax(pb, axis=1)]
    b_auc.append(roc_auc_score(yb, pb, multi_class="ovr", average="macro", labels=classes))
    b_f1.append(f1_score(yb, pb_pred, average="macro", labels=classes, zero_division=0))

auc_ci = np.percentile(b_auc, [2.5, 97.5])
f1_ci = np.percentile(b_f1, [2.5, 97.5])

out = pd.DataFrame([
    {"metric": primary, "value": auc, "ci_low": auc_ci[0], "ci_high": auc_ci[1]},
    {"metric": "f1_macro", "value": f1, "ci_low": f1_ci[0], "ci_high": f1_ci[1]},
]).round(4)
mlca_runtime.save_table(out, "test_metrics.csv", caption="Frozen model test metrics with 95% bootstrap CI")

mlca_runtime.record_metric(primary, float(auc), "test", ci=(float(auc_ci[0]), float(auc_ci[1])))
mlca_runtime.record_metric("f1_macro", float(f1), "test", ci=(float(f1_ci[0]), float(f1_ci[1])))
mlca_runtime.record_fact_candidate("test_roc_auc", float(auc), source="test_metrics.csv")

print(f"test rows={len(test)} features={len(feature_cols)} selected={best_key.split('/')[-1]}")
print(f"{primary}={auc:.4f} CI=({auc_ci[0]:.4f},{auc_ci[1]:.4f})")
print(f"f1_macro={f1:.4f} CI=({f1_ci[0]:.4f},{f1_ci[1]:.4f})")
