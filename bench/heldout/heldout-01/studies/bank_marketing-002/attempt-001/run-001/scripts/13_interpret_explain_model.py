import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
task = protocol["task"]
metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
print("train", train.shape, "val", val.shape)

def keylike(col, s):
    if s.dtype == object or str(s.dtype).startswith("str"):
        return False
    if s.nunique(dropna=True) == len(s) and not np.issubdtype(s.dtype, np.number):
        return True
    if np.issubdtype(s.dtype, np.integer):
        return s.nunique(dropna=True) == len(s)
    return False

drop = set(targets) | set(excluded) | ({group} if group else set())
feats = []
for c in train.columns:
    if c in drop:
        continue
    if keylike(c, train[c]):
        continue
    feats.append(c)
print("n_features", len(feats))

with open(inputs["08_model_train_logistic_Y7873DWF/model.pkl"], "rb") as f:
    model = pickle.load(f)

y_target = targets[0]
X_val = val[feats].copy()
y_val = val[y_target]
if y_val.dtype == object:
    y_val = y_val.map({"no": 0, "yes": 1}).astype(int)

if hasattr(model, "feature_names_in_"):
    cols = list(model.feature_names_in_)
    X_val = X_val[cols]

scoring = "roc_auc" if task["type"] == "binary" else "roc_auc_ovr"
res = permutation_importance(model, X_val, y_val, scoring=scoring,
                             n_repeats=20, random_state=42, n_jobs=1)

names = list(X_val.columns)
imp = pd.DataFrame({
    "feature": names,
    "importance_mean": np.round(res.importances_mean, 6),
    "importance_sd": np.round(res.importances_std, 6),
}).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)

mlca_runtime.save_table(imp, "interpretation.csv", caption="Permutation importance (top 15)")
top = str(imp.iloc[0]["feature"])
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
print("top_feature", top)
