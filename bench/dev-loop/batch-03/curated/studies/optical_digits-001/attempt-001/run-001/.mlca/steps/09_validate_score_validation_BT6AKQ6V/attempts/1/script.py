import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column", None)

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feat_cols = [c for c in train.columns if c not in drop]
if not feat_cols:
    raise ValueError("No feature columns after exclusions")

with open(inputs["05_preprocess_fit_preprocessing_NFXS64E4/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_HGAM38DP/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def contains(pipe, obj):
    if not isinstance(pipe, Pipeline):
        return False
    return any(step is obj for step in pipe.named_steps.values())

if isinstance(sel, Pipeline) and contains(sel, pre):
    frontend = sel
else:
    frontend = Pipeline([("pre", pre), ("sel", sel)])

model_paths = {
    "baseline": "07_model_train_baseline_MYG51GQ8/model.pkl",
    "alternative": "08_model_train_alternative_F57HEDMY/alternative_model.pkl",
    "latest": "latest_model",
}
models = {}
for name, key in model_paths.items():
    with open(inputs[key], "rb") as f:
        models[name] = pickle.load(f)

X_val = val[feat_cols]
X_val_scaled = np.asarray(frontend.transform(X_val))

rows = []
for name, model in models.items():
    if isinstance(model, Pipeline) and (contains(model, pre) or contains(model, sel)):
        pred = np.asarray(model.predict(X_val))
    else:
        pred = np.asarray(model.predict(X_val_scaled))
    mse = float(np.mean((pred - X_val_scaled) ** 2))
    rows.append({"model": name, "metric": "reconstruction_error", "value": mse})

df = pd.DataFrame(rows)
best = df.loc[df["value"].idxmin()]
mlca_runtime.save_table(df, "validation_metrics.csv",
                        caption="Validation reconstruction error in scaled space")
mlca_runtime.record_metric("reconstruction_error", float(best["value"]), "val")
mlca_runtime.record_fact_candidate("best_validation_score", float(best["value"]),
                                   source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", str(best["model"]),
                                   source="validation_metrics.csv")

print(f"val rows={len(val)} features={len(feat_cols)}")
print(f"best={best['model']} reconstruction_error={best['value']:.6f}")
