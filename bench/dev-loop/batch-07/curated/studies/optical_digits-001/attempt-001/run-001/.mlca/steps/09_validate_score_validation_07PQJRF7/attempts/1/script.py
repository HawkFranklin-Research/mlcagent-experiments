import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
primary_metric = protocol["task"]["primary_metric"]
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_keylike(s):
    if s.dtype.kind in "iu" and s.nunique() == len(s):
        return True
    if s.dtype == object and s.nunique() == len(s):
        return True
    return False

feat_cols = [c for c in train.columns
             if c not in targets and c not in excluded and c != group_col
             and not is_keylike(train[c])]
X_val = val[feat_cols].copy()

with open(inputs["05_preprocess_fit_preprocessing_E2NYE0HC/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_8M4G64VT/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def build_frontend(pre, sel):
    if isinstance(sel, Pipeline) and len(sel.named_steps) > 1:
        return sel
    if isinstance(pre, Pipeline):
        return Pipeline(pre.steps + [("selector", sel)])
    return Pipeline([("pre", pre), ("selector", sel)])

frontend = build_frontend(pre, sel)
Xf_val = frontend.transform(X_val)
print("frontend shape", Xf_val.shape)

def get_Xf_and_est(model, X_raw, frontend):
    if isinstance(model, Pipeline):
        names = list(model.named_steps.keys())
        est = model.named_steps[names[-1]]
        if len(names) > 1:
            Xf = Pipeline(model.steps[:-1]).transform(X_raw)
        else:
            Xf = frontend.transform(X_raw)
    else:
        est = model
        Xf = frontend.transform(X_raw)
    return Xf, est

def recon(est, Xf):
    if isinstance(est, PCA):
        return est.inverse_transform(est.transform(Xf))
    return est.predict(Xf)

candidates = [
    ("baseline", inputs["07_model_train_baseline_JYP4DTKE/model.pkl"]),
    ("alternative", inputs["08_model_train_alternative_29T9VJ6F/alternative_model.pkl"]),
]

rows = []
for name, path in candidates:
    with open(path, "rb") as f:
        model = pickle.load(f)
    Xf, est = get_Xf_and_est(model, X_val, frontend)
    Xr = recon(est, Xf)
    mse = float(np.mean((Xf - Xr) ** 2))
    rows.append({"model": name, "metric": primary_metric, "value": mse})
    print(name, est.__class__.__name__, Xf.shape, round(mse, 4))

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation reconstruction error")
best = df.loc[df["value"].idxmin()]
mlca_runtime.record_metric(primary_metric, float(best["value"]), "val")
mlca_runtime.record_fact_candidate("best_validation_score", float(best["value"]), source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", str(best["model"]), source="validation_metrics.csv")
print("best", best["model"], round(float(best["value"]), 4))
