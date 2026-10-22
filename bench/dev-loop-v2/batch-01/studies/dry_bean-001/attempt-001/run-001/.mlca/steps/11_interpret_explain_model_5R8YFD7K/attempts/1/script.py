import json
import pickle
import pandas as pd
from sklearn.inspection import permutation_importance
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
target = targets[0]
primary = protocol["task"]["primary_metric"]

val = pd.read_parquet(inp["val"])

with open(inp["latest_model"], "rb") as f:
    model = pickle.load(f)
if not hasattr(model, "predict_proba"):
    for key in [
        "08_model_train_alternative_JA5Q5D9S/alternative_model.pkl",
        "07_model_train_logistic_0BAYN51T/model.pkl",
    ]:
        with open(inp[key], "rb") as f:
            candidate = pickle.load(f)
        if hasattr(candidate, "predict_proba"):
            model = candidate
            break

if hasattr(model, "feature_names_in_"):
    features = list(model.feature_names_in_)
elif hasattr(model, "named_steps"):
    features = None
    for step in model.named_steps.values():
        if hasattr(step, "feature_names_in_"):
            features = list(step.feature_names_in_)
            break
    if features is None:
        raise ValueError("Loaded model has no feature_names_in_ and no nested feature schema.")
else:
    raise ValueError("Loaded model has no feature_names_in_ and is not a Pipeline.")

X = val[features]
y = val[target]

result = permutation_importance(
    model,
    X,
    y,
    scoring=primary,
    n_repeats=20,
    random_state=42,
    n_jobs=1,
)

interp = pd.DataFrame(
    {
        "feature": features,
        "importance_mean": result.importances_mean,
        "importance_sd": result.importances_std,
    }
).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)

def sig4(x):
    return float(f"{x:.4g}")

interp["importance_mean"] = interp["importance_mean"].map(sig4)
interp["importance_sd"] = interp["importance_sd"].map(sig4)

mlca_runtime.save_table(
    interp,
    "interpretation.csv",
    caption="Permutation importance on validation rows",
)

top_feature = interp.loc[0, "feature"] if len(interp) else None
mlca_runtime.record_fact_candidate(
    "top_feature",
    top_feature,
    source="interpretation.csv",
)

print(f"val rows={len(val)}, features={len(features)}")
print(f"top feature={top_feature}, importance={interp.loc[0, 'importance_mean'] if len(interp) else None}")
