import json, pickle
import numpy as np, pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
IN = mlca_runtime.inputs()
protocol = json.loads(IN["_protocol"].read_text())
task = protocol["task"]
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")
primary = task["primary_metric"]

train = pd.read_parquet(IN["train"])
test = pd.read_parquet(IN["test"])

def is_keylike(s):
    try:
        return bool(s.notna().all() and s.is_unique and (s.dtype.kind in "iu" or s.dtype == object))
    except Exception:
        return False

blocked = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in blocked and not is_keylike(train[c])]

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load(IN["05_preprocess_fit_preprocessing_DYJSGEW8/preprocessor.pkl"])
sel = load(IN["06_features_select_features_KAD36QCW/selector.pkl"])
mdl = load(IN["latest_model"])

names = list(mdl.named_steps) if isinstance(mdl, Pipeline) else []
steps = []
if not any("preprocess" in n for n in names):
    steps.append(("preprocessor", pre))
if not any("selector" in n for n in names):
    steps.append(("selector", sel))
steps += list(mdl.steps) if isinstance(mdl, Pipeline) else [("model", mdl)]
full = Pipeline(steps)

first = full.steps[0][1]
cols = list(first.feature_names_in_) if hasattr(first, "feature_names_in_") else feature_cols
X = test.loc[:, cols]
y_true = test[targets[0]].to_numpy()
classes = np.asarray(full.classes_)
proba = full.predict_proba(X)

def compute(yt, yp):
    a = roc_auc_score(yt, yp, multi_class="ovr", average="macro", labels=classes)
    f = f1_score(yt, classes[yp.argmax(1)], average="macro", labels=classes, zero_division=0)
    return a, f

auc, mf1 = compute(y_true, proba)
rng = np.random.default_rng(42)
boot = {"roc_auc_ovr": [], "macro_f1": []}
n = len(y_true)
for _ in range(1000):
    i = rng.integers(0, n, n)
    try:
        a, f = compute(y_true[i], proba[i])
    except Exception:
        continue
    boot["roc_auc_ovr"].append(a)
    boot["macro_f1"].append(f)

def ci(vals):
    arr = np.asarray(vals)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

rows = []
for name, val in [("roc_auc_ovr", auc), ("macro_f1", mf1)]:
    lo, hi = ci(boot[name])
    rows.append({"metric": name, "value": round(val, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
res = pd.DataFrame(rows)
mlca_runtime.save_table(res, "test_metrics.csv")

main = res.loc[res["metric"] == primary].iloc[0]
mlca_runtime.record_metric(primary, float(main["value"]), "test", ci=[float(main["ci_low"]), float(main["ci_high"])])
if primary in ("roc_auc", "roc_auc_ovr"):
    mlca_runtime.record_fact_candidate("test_roc_auc", float(main["value"]), unit="auc", source="test_metrics.csv")

print("test rows:", len(test), "features:", len(cols))
print(primary, round(float(main["value"]), 4), "ci", [round(float(main["ci_low"]), 4), round(float(main["ci_high"]), 4)])
