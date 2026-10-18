import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column", None)
primary_metric = protocol["task"]["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

feat_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    if train[c].dtype == object and train[c].nunique() == len(train) and train[c].notna().all():
        continue
    feat_cols.append(c)

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

if hasattr(model, "feature_names_in_"):
    cols = list(model.feature_names_in_)
elif hasattr(model, "steps") and hasattr(model.steps[0][1], "feature_names_in_"):
    cols = list(model.steps[0][1].feature_names_in_)
else:
    cols = feat_cols

if primary_metric == "roc_auc":
    def scorer(est, X, y):
        proba = est.predict_proba(X)[:, 1]
        return roc_auc_score(y, proba)
elif primary_metric == "roc_auc_ovr":
    def scorer(est, X, y):
        proba = est.predict_proba(X)
        return roc_auc_score(y, proba, multi_class="ovr", average="macro")
else:
    raise ValueError(f"Unsupported primary_metric: {primary_metric}")

y_val = val[targets[0]]
X_val = val[cols]

result = permutation_importance(
    model, X_val, y_val, scoring=scorer,
    n_repeats=20, random_state=42, n_jobs=1
)

imp = pd.DataFrame({
    "feature": cols,
    "importance_mean": result.importances_mean,
    "importance_sd": result.importances_std,
}).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)

imp["importance_mean"] = imp["importance_mean"].map(lambda x: float(f"{x:.4g}"))
imp["importance_sd"] = imp["importance_sd"].map(lambda x: float(f"{x:.4g}"))

mlca_runtime.save_table(imp, "interpretation.csv")
mlca_runtime.record_fact_candidate("top_feature", imp.iloc[0]["feature"], source="interpretation.csv")
print(f"features: {len(cols)}, repeats: 20, top: {imp.iloc[0]['feature']}")
