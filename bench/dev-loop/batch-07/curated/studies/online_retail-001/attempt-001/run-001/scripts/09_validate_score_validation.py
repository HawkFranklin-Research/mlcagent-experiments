import json, pickle
import numpy as np
import pandas as pd
from scipy.sparse import issparse
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
task = proto["task"]
targets = list(proto.get("targets") or [])
excluded = list(proto.get("excluded_columns") or [])
group_col = proto.get("group_column")
primary = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def keylike(s):
    return s.dtype.kind in "iu" and s.is_unique

feat_cols = [c for c in train.columns
             if c not in targets and c not in excluded and c != group_col
             and not keylike(train[c])]
X_val = val[feat_cols]

with open(inputs["05_preprocess_fit_preprocessing_X6XZMNTX/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_79PZREQ1/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

if isinstance(sel, Pipeline):
    fe = sel
else:
    steps = list(pre.steps) if isinstance(pre, Pipeline) else [("pre", pre)]
    fe = Pipeline(steps + [("sel", sel)])

Xv_t = fe.transform(X_val)
if issparse(Xv_t):
    Xv_t = Xv_t.toarray()
Xv_t = np.asarray(Xv_t)

with open(inputs["07_model_train_baseline_7R0X4ZSD/model.pkl"], "rb") as f:
    cands = {"baseline": pickle.load(f)}
with open(inputs["08_model_train_alternative_3C6W5R97/alternative_model.pkl"], "rb") as f:
    cands["alternative"] = pickle.load(f)

direction = {"silhouette": "max", "calinski_harabasz": "max", "davies_bouldin": "min"}
rows, prim = [], {}
n = min(10000, len(Xv_t))
for name, m in cands.items():
    labels = np.asarray(m.predict(X_val))
    if len(np.unique(labels)) < 2:
        continue
    sil = float(silhouette_score(Xv_t, labels, sample_size=n, random_state=42))
    ch = float(calinski_harabasz_score(Xv_t, labels))
    db = float(davies_bouldin_score(Xv_t, labels))
    rows += [{"model": name, "metric": "silhouette", "value": sil},
             {"model": name, "metric": "calinski_harabasz", "value": ch},
             {"model": name, "metric": "davies_bouldin", "value": db}]
    prim[name] = {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}[primary]

metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation clustering metrics")

ord_fn = max if direction.get(primary, "max") == "max" else min
best = ord_fn(prim, key=prim.get)
mlca_runtime.record_fact_candidate("best_model", best, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", float(prim[best]), source="validation_metrics.csv")
mlca_runtime.record_metric(primary, float(prim[best]), split="val")
print(metrics_df.shape, "best:", best, primary, "=", round(prim[best], 4))
