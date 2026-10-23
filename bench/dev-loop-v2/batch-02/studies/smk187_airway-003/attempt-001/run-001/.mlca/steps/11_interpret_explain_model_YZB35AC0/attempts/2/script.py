import json, pickle
import numpy as np, pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for k in ["targets", "excluded_columns", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing protocol key {k!r}; available: {list(protocol)}")
targets = protocol["targets"]
metric = protocol["task"]["primary_metric"]

train_y = pd.read_parquet(inputs["train"], columns=targets)
val_y = pd.read_parquet(inputs["val"], columns=targets)
print(f"train rows={len(train_y)}, val rows={len(val_y)}")

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

def find_est(obj, depth=0):
    if depth > 8 or obj is None:
        return None
    if hasattr(obj, "coef_") or hasattr(obj, "feature_importances_"):
        return obj
    buckets = []
    if hasattr(obj, "steps"):
        buckets.append([s[1] for s in obj.steps])
    if hasattr(obj, "named_steps"):
        buckets.append(list(obj.named_steps.values()))
    for attr in ["calibrated_classifiers_", "estimators_", "estimator"]:
        if hasattr(obj, attr):
            v = getattr(obj, attr)
            buckets.append(v if isinstance(v, list) else [v])
    for bucket in buckets:
        for sub in reversed(list(bucket)):
            r = find_est(sub, depth + 1)
            if r is not None:
                return r
    return None

est = find_est(model)
n_ranked = 0
if est is not None:
    if hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        importance = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef).ravel()
        method = "abs_coefficient_mean"
    else:
        importance = np.asarray(est.feature_importances_)
        method = "feature_importances"
    n_imp = int(len(importance))
    n_ranked = n_imp
    candidates = []
    for src in [est, model]:
        if hasattr(src, "feature_names_in_"):
            candidates.append(list(src.feature_names_in_))
    if hasattr(model, "steps"):
        try:
            candidates.append(list(model[:-1].get_feature_names_out()))
        except Exception:
            pass
    feature_names = next((c for c in candidates if len(c) == n_imp), None)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_imp)]
    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": importance.astype(float),
        "importance_sd": np.nan,
    })
    df = df.sort_values("importance_mean", ascending=False, kind="mergesort").head(15).reset_index(drop=True)
    df["importance_mean"] = df["importance_mean"].map(lambda v: float(f"{v:.4g}"))
    caption = f"Top 15 frozen features by {method}; metric={metric}; no refit."
else:
    method = "unavailable_limitation"
    df = pd.DataFrame({
        "feature": ["LIMITATION_no_reliable_importance_ranking"],
        "importance_mean": [np.nan],
        "importance_sd": [np.nan],
    })
    caption = "Frozen model exposes no reliable coefficient or feature_importances_ ranking."

mlca_runtime.save_table(df, "interpretation.csv", caption=caption)
mlca_runtime.record_metric("n_ranked_features", float(n_ranked), "val")
if est is not None:
    mlca_runtime.record_fact_candidate("top_feature", str(df.loc[0, "feature"]), source="interpretation.csv")
    mlca_runtime.record_metric("top_feature_importance", float(df.loc[0, "importance_mean"]), "val")
print(f"method={method} ranked={n_ranked} top={df.loc[0, 'feature']}")
