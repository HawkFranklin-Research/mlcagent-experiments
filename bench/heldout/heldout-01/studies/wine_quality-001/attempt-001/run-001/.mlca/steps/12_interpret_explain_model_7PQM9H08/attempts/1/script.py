import json, pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

protocol = json.loads(inputs["_protocol"].read_text())
task = protocol["task"]
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def feature_cols(df):
    cols = [c for c in df.columns if c not in targets and c not in excluded and c != group]
    out_cols = []
    n = len(df)
    for c in cols:
        s = df[c]
        if s.dtype == object or str(s.dtype).startswith("string"):
            if s.nunique() == n:
                continue
        elif pd.api.types.is_integer_dtype(s):
            if s.nunique() == n and s.nunique() > 20:
                continue
        out_cols.append(c)
    return out_cols

feats = feature_cols(train)
print("train", train.shape, "val", val.shape, "nfeat", len(feats))

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

expected = list(getattr(model, "feature_names_in_", feats))
Xval = val[expected]
yval = val[targets[0]]
print("model features", len(expected))

scoring = "roc_auc_ovr" if metric in ("roc_auc_ovr", "roc_auc") else "roc_auc_ovr"
if task["type"] == "binary":
    scoring = "roc_auc"

r = permutation_importance(model, Xval, yval, scoring=scoring,
                           n_repeats=20, random_state=42, n_jobs=1)

imp = pd.DataFrame({
    "feature": expected,
    "importance_mean": np.round(r.importances_mean, 4),
    "importance_sd": np.round(r.importances_std, 4),
}).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)

mlca_runtime.save_table(imp, "interpretation.csv", caption="Permutation importance on validation rows")
top = imp.iloc[0]["feature"]
mlca_runtime.record_fact_candidate("top_feature", str(top), source="interpretation.csv")
print("top_feature", top, "score", float(r.importances_mean.max()))
