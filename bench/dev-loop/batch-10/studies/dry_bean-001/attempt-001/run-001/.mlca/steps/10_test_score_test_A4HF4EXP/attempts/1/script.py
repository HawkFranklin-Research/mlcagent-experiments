import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
task = protocol["task"]
primary_metric = task["primary_metric"]
target = protocol["targets"][0]

test = pd.read_parquet(inp["test"])

logistic_path = inp["07_model_train_logistic_Z7HS8D6N/model.pkl"]
alt_path = inp["08_model_train_alternative_MG1H66HT/alternative_model.pkl"]
vm = pd.read_csv(inp["09_validate_score_validation_YD98GHNZ/validation_metrics.csv"])

selected = None
if {"metric", "value"}.issubset(vm.columns):
    sub = vm[vm["metric"] == primary_metric]
    cand_col = next((c for c in ("candidate", "model", "candidate_id", "model_id") if c in sub.columns), None)
    if cand_col is not None and len(sub):
        selected = str(sub.sort_values("value").iloc[-1][cand_col]).lower()

if selected is not None:
    model_path = alt_path if "alt" in selected else logistic_path
else:
    model_path = inp["latest_model"]

with open(model_path, "rb") as f:
    model = pickle.load(f)

feats = list(model.feature_names_in_)
X_test = test[feats]
y_test = test[target].to_numpy()
classes = np.asarray(model.classes_)
proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

def macro_ovr(y, p):
    pres = [c for c in classes if c in set(y)]
    if len(pres) == len(classes):
        return roc_auc_score(y, p, labels=classes, multi_class="ovr", average="macro")
    vals = [roc_auc_score((y == c).astype(int), p[:, list(classes).index(c)]) for c in pres]
    return float(np.mean(vals))

auc = macro_ovr(y_test, proba)
f1 = f1_score(y_test, y_pred, average="macro")

rng = np.random.default_rng(42)
n = len(test)
auc_bs, f1_bs = [], []
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yb, pb, ypb = y_test[idx], proba[idx], y_pred[idx]
    try:
        auc_bs.append(macro_ovr(yb, pb))
    except Exception:
        pass
    try:
        f1_bs.append(f1_score(yb, ypb, average="macro"))
    except Exception:
        pass

auc_lo, auc_hi = np.percentile(auc_bs, [2.5, 97.5])
f1_lo, f1_hi = np.percentile(f1_bs, [2.5, 97.5])

out_df = pd.DataFrame([
    {"metric": primary_metric, "value": auc, "ci_low": auc_lo, "ci_high": auc_hi},
    {"metric": "macro_f1", "value": f1, "ci_low": f1_lo, "ci_high": f1_hi},
])
mlca_runtime.save_table(out_df, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

mlca_runtime.record_metric(primary_metric, float(auc), "test", ci=(float(auc_lo), float(auc_hi)))
mlca_runtime.record_metric("macro_f1", float(f1), "test", ci=(float(f1_lo), float(f1_hi)))
mlca_runtime.record_fact_candidate("test_roc_auc", float(auc), source="test_metrics.csv")

print(f"model: {model_path.name}, n_classes: {len(classes)}, test_rows: {n}")
print(f"{primary_metric}={auc:.4f} CI=({auc_lo:.4f},{auc_hi:.4f})")
print(f"macro_f1={f1:.4f} CI=({f1_lo:.4f},{f1_hi:.4f})")
