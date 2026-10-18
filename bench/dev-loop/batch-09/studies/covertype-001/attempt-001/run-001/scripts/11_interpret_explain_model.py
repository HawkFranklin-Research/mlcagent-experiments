import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

SEED = 42
mlca_runtime.set_seed(SEED)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = set(protocol.get("excluded_columns", []))
group_col = protocol.get("group_column")
task = protocol.get("task", {}) or {}
metric = task.get("primary_metric", "roc_auc_ovr")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)


def _auc(est, X, y):
    p = est.predict_proba(X)
    classes = getattr(est, "classes_", None)
    if classes is None:
        classes = est[-1].classes_
    return roc_auc_score(y, p, multi_class="ovr", average="macro", labels=classes)


cand = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
feat_cols = []
for c in cand:
    s = train[c]
    if pd.api.types.is_integer_dtype(s) and s.nunique() == len(s) and len(s) > 2:
        continue
    if s.dtype == object and s.nunique() == len(s):
        continue
    feat_cols.append(c)

if hasattr(model, "feature_names_in_"):
    mcols = list(model.feature_names_in_)
    missing = [c for c in mcols if c not in val.columns]
    if missing:
        raise RuntimeError(f"Model expects columns missing from val: {missing}")
    feat_cols = mcols

X_val = val[feat_cols].copy()
y_val = val[targets[0]].copy()

if len(X_val) > 10000:
    idx = np.random.RandomState(SEED).choice(len(X_val), size=10000, replace=False)
    X_val = X_val.iloc[idx].reset_index(drop=True)
    y_val = y_val.iloc[idx].reset_index(drop=True)

res = permutation_importance(
    model, X_val, y_val, scoring=_auc, n_repeats=20,
    random_state=SEED, n_jobs=1,
)

imp = pd.DataFrame({
    "feature": list(X_val.columns),
    "importance_mean": np.round(res.importances_mean, 4),
    "importance_sd": np.round(res.importances_std, 4),
}).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)

mlca_runtime.save_table(
    imp, "interpretation.csv",
    caption=f"Permutation importance (top 15, val, 20 repeats, metric={metric})",
)

top_feature = str(imp.iloc[0, 0])
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"val rows used: {len(X_val)}; features scored: {len(feat_cols)}; metric: {metric}")
print(f"top feature: {top_feature} (mean {imp.iloc[0, 1]})")
