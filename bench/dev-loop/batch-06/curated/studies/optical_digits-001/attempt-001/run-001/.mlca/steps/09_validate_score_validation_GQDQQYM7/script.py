import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
metric_name = protocol["task"]["primary_metric"]
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feat_cols = [c for c in train.columns if c not in drop]
for c in list(feat_cols):
    s = train[c]
    if s.dtype == object or (pd.api.types.is_integer_dtype(s) and s.nunique() == len(s)):
        feat_cols.remove(c)
X_val = val[feat_cols]

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load(inputs["05_preprocess_fit_preprocessing_RDQ93D4J/preprocessor.pkl"])
sel = load(inputs["06_features_retain_features_0YD36CMR/selector_all.pkl"])
m1 = load(inputs["07_model_train_baseline_EE9J97EQ/model.pkl"])
m2 = load(inputs["08_model_train_alternative_N9ZERH70/alternative_model.pkl"])

def build_frontend(pre, sel):
    if isinstance(sel, Pipeline) and len(sel.steps) > 1:
        return sel
    if isinstance(pre, Pipeline) and len(pre.steps) > 1 and 'select' in pre.steps[-1][0].lower():
        return pre
    return Pipeline([('pre', pre), ('sel', sel)])

def split(m):
    if isinstance(m, Pipeline) and len(m.steps) > 1:
        steps = list(m.steps)
        return Pipeline(steps[:-1]), steps[-1][1]
    est = m.steps[-1][1] if isinstance(m, Pipeline) else m
    return build_frontend(pre, sel), est

def recon_err(front, est, X):
    Xf = front.transform(X)
    if isinstance(est, PCA):
        rec = est.inverse_transform(est.transform(Xf))
    else:
        rec = est.predict(Xf)
    return float(np.mean((np.asarray(Xf) - np.asarray(rec)) ** 2))

rows = []
for name, m in [("baseline", m1), ("alternative", m2)]:
    front, est = split(m)
    rows.append({"model": name, "metric": metric_name, "value": recon_err(front, est, X_val)})

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "validation_metrics.csv", "Validation reconstruction error")

best = df.loc[df["value"].idxmin()]
mlca_runtime.record_metric(metric_name, float(best["value"]), "val")
mlca_runtime.record_fact_candidate("best_validation_score", float(best["value"]), source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", str(best["model"]), source="validation_metrics.csv")

print(f"val rows: {len(X_val)}, features: {len(feat_cols)}")
print(f"baseline err: {rows[0]['value']:.4g}")
print(f"alternative err: {rows[1]['value']:.4g}")
print(f"best: {best['model']} = {best['value']:.4g}")
