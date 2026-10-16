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

X_val = val[feat_cols]

model_paths = {
    "baseline": "07_model_train_baseline_MYG51GQ8/model.pkl",
    "alternative": "08_model_train_alternative_F57HEDMY/alternative_model.pkl",
    "latest": "latest_model",
}

rows = []
for name, key in model_paths.items():
    with open(inputs[key], "rb") as f:
        model = pickle.load(f)
    if not isinstance(model, Pipeline) or len(model.steps) < 2:
        raise ValueError(f"Model {name} is not a complete pipeline")
    target = model[:-1].transform(X_val)
    pred = model.predict(X_val)
    mse = float(np.mean((np.asarray(pred) - np.asarray(target)) ** 2))
    rows.append({"model": name, "metric": "reconstruction_error", "value": mse})

df = pd.DataFrame(rows)
df["value"] = df["value"].map(lambda x: float(f"{x:.4g}"))
best = df.loc[df["value"].idxmin()]
mlca_runtime.save_table(df, "validation_metrics.csv",
                        caption="Validation reconstruction error in scaled space")
mlca_runtime.record_metric("reconstruction_error", float(best["value"]), "val")
mlca_runtime.record_fact_candidate("best_validation_score", float(best["value"]),
                                   source="validation_metrics.csv")

print(f"val rows={len(val)} features={len(feat_cols)}")
print(f"best={best['model']} reconstruction_error={best['value']:.6f}")
