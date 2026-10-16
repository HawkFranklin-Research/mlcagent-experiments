import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_poisson_deviance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

proto = json.loads(inputs["_protocol"].read_text())
targets = proto["targets"]
excluded = proto.get("excluded_columns", [])
group_col = proto.get("group_column")
task = proto["task"]
primary = task.get("primary_metric", "poisson_deviance")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_keylike(s):
    n = s.nunique(dropna=True)
    if n != len(s) or n <= 1:
        return False
    if pd.api.types.is_numeric_dtype(s):
        return pd.api.types.is_integer_dtype(s)
    return True

feat_cols = [c for c in train.columns
             if c not in targets and c not in excluded and c != group_col
             and not is_keylike(train[c])]

y_val = val[targets]

models = {
    "baseline": inputs["07_model_train_baseline_7Z79BW5N/model.pkl"],
    "alternative": inputs["08_model_train_alternative_PY0Z053P/alternative_model.pkl"],
}

rows = []
scores = {}
for name, path in models.items():
    with open(path, "rb") as f:
        model = pickle.load(f)
    cols = list(getattr(model, "feature_names_in_", feat_cols))
    Xv = val[cols]
    pred = np.asarray(model.predict(Xv))
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    per = {}
    for i, t in enumerate(targets):
        yt = y_val[t].to_numpy()
        pt = pred[:, i]
        rmse = float(np.sqrt(mean_squared_error(yt, pt)))
        mae = float(mean_absolute_error(yt, pt))
        r2 = float(r2_score(yt, pt))
        pdv = float(mean_poisson_deviance(yt, np.clip(pt, 1e-9, None)))
        per[t] = {"rmse": rmse, "mae": mae, "r2": r2, "poisson_deviance": pdv}
        for k, v in per[t].items():
            rows.append({"model": name, "metric": f"{k}_{t}", "value": v})
    macro = {k: float(np.mean([per[t][k] for t in targets]))
             for k in ["rmse", "mae", "r2", "poisson_deviance"]}
    for k, v in macro.items():
        rows.append({"model": name, "metric": f"{k}_macro", "value": v})
    scores[name] = macro[primary]

best = min(scores, key=scores.get)
best_score = scores[best]
best_rmse = float(np.mean([r["value"] for r in rows
                           if r["model"] == best and r["metric"] == "rmse_macro"]))

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "validation_metrics.csv", "Validation metrics per model and target")

with open(models[best], "rb") as f:
    best_model = pickle.load(f)
mlca_runtime.save_model(best_model, "frozen_validation_candidate.pkl")

mlca_runtime.record_metric("rmse", best_rmse, "val")
mlca_runtime.record_fact_candidate("best_validation_score", best_score,
                                   source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best, source="validation_metrics.csv")

print(f"val rows={len(val)} features={len(feat_cols)} targets={targets}")
print(f"primary={primary} scores={scores}")
print(f"best={best} score={best_score:.4f} rmse={best_rmse:.4f}")
