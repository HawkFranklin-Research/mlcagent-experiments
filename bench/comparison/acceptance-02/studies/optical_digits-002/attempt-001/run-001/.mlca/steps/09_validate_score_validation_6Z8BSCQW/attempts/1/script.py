import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
try:
    primary_metric = protocol["task"]["primary_metric"]
except KeyError as exc:
    raise KeyError(f"Missing task.primary_metric; top-level keys: {list(protocol)}") from exc

targets = set(protocol.get("targets", []))
excluded = set(protocol.get("excluded_columns", []))
group_col = protocol.get("group_column")
exclude_cols = targets | excluded | ({group_col} if group_col else set())

val = pd.read_parquet(inputs["val"])

def default_features(frame):
    feats = [c for c in frame.columns if c not in exclude_cols]
    return [
        c for c in feats
        if not (pd.api.types.is_object_dtype(frame[c]) and frame[c].nunique(dropna=True) == len(frame))
    ]

default_feats = default_features(val)

def model_features(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def load_pkl(path):
    return pickle.loads(path.read_bytes())

def recon_mse(model, X):
    if isinstance(model, Pipeline):
        front = Pipeline(model.steps[:-1]) if len(model.steps) > 1 else None
        est = model.steps[-1][1]
    else:
        front, est = None, model
    Xf = front.transform(X) if front is not None else X
    if hasattr(Xf, "toarray"):
        Xf = Xf.toarray()
    Xa = np.asarray(Xf)
    if hasattr(est, "transform") and hasattr(est, "inverse_transform"):
        rec = est.inverse_transform(est.transform(Xf))
    elif hasattr(est, "predict"):
        rec = est.predict(Xf)
    else:
        raise TypeError(f"Unsupported reconstruction estimator: {type(est).__name__}")
    return float(np.mean((np.asarray(rec) - Xa) ** 2))

candidates = [
    ("07_model_train_baseline_EGH0EDWF", load_pkl(inputs["07_model_train_baseline_EGH0EDWF/model.pkl"])),
    ("08_model_train_alternative_X1KB5C0A", load_pkl(inputs["08_model_train_alternative_X1KB5C0A/alternative_model.pkl"])),
]

rows = []
scores = {}
for name, model in candidates:
    feats = model_features(model) or default_feats
    X = val.loc[:, feats]
    score = recon_mse(model, X)
    scores[name] = score
    rows.append({"model": name, "metric": primary_metric, "value": float(f"{score:.4g}")})

best_name = min(scores, key=scores.get)
best_score = float(f"{scores[best_name]:.4g}")
metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation reconstruction errors; lower is better.")

mlca_runtime.record_metric(primary_metric, best_score, split="val")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")

print(f"val rows: {len(val)}, candidates: {len(candidates)}")
print(f"best {primary_metric}: {best_score:.6g} ({best_name})")
