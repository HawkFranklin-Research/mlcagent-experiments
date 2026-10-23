import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

with open(inputs["_protocol"], "r") as f:
    protocol = json.load(f)
for k in ["targets", "excluded_columns", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing protocol key {k!r}; available keys: {list(protocol)}")
targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
metric = protocol["task"]["primary_metric"]

train_y = pd.read_parquet(inputs["train"], columns=targets)
val_y = pd.read_parquet(inputs["val"], columns=targets)
print(f"train rows={len(train_y)}, val rows={len(val_y)}")

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

est = model.steps[-1][1] if hasattr(model, "steps") else model

raw_names = None
for obj in [model] + ([s[1] for s in model.steps] if hasattr(model, "steps") else []):
    if hasattr(obj, "feature_names_in_"):
        raw_names = list(obj.feature_names_in_)
        break
if raw_names is None:
    raw_names = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None

if hasattr(est, "feature_names_in_"):
    feature_names = list(est.feature_names_in_)
elif hasattr(model, "steps"):
    feature_names = list(model[:-1].get_feature_names_out())
elif raw_names is not None:
    feature_names = raw_names
else:
    raise RuntimeError("Cannot determine model feature names.")

if hasattr(est, "coef_"):
    coef = np.asarray(est.coef_)
    importance = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef).ravel()
    method = "abs_coefficient_mean"
elif hasattr(est, "feature_importances_"):
    importance = np.asarray(est.feature_importances_)
    method = "feature_importances"
else:
    raise RuntimeError("Frozen model has neither coef_ nor feature_importances_.")

if len(feature_names) != len(importance):
    raise RuntimeError(
        f"Feature-name/importance length mismatch: {len(feature_names)} vs {len(importance)}"
    )

n_input = len(raw_names) if raw_names is not None else len(importance)
out = pd.DataFrame(
    {
        "feature": feature_names,
        "importance_mean": importance,
        "importance_sd": np.nan,
    }
)
out = out.sort_values("importance_mean", ascending=False, kind="mergesort").head(15).reset_index(drop=True)
out["importance_mean"] = out["importance_mean"].map(lambda v: float(f"{v:.4g}"))

mlca_runtime.save_table(
    out,
    "interpretation.csv",
    caption=f"Top 15 frozen model features by {method}; metric={metric}; no refit.",
)
mlca_runtime.record_fact_candidate(
    "top_feature", str(out.loc[0, "feature"]), source="interpretation.csv"
)
mlca_runtime.record_metric("n_input_features", float(n_input), "val")
mlca_runtime.record_metric("n_ranked_features", float(len(importance)), "val")
mlca_runtime.record_metric("top_feature_importance", float(out.loc[0, "importance_mean"]), "val")

print(f"method={method} input_features={n_input} ranked_features={len(importance)} repeats=0")
print(f"top feature={out.loc[0, 'feature']} importance={out.loc[0, 'importance_mean']:.4g}")
