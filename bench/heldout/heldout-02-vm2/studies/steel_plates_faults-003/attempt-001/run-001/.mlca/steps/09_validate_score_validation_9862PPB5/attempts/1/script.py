import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
available = list(proto.keys())

def need(key):
    if key not in proto:
        raise KeyError(f"Missing {key}; available top-level keys: {available}")
    return proto[key]

targets = need("targets")
excluded = need("excluded_columns")
task = need("task")
primary_metric = task["primary_metric"]
group_col = proto.get("group_column")

val = pd.read_parquet(inputs["val"])
missing_targets = [t for t in targets if t not in val.columns]
if missing_targets:
    raise KeyError(f"Missing target columns in val: {missing_targets}")

def feature_names(obj):
    if getattr(obj, "feature_names_in_", None) is not None:
        return list(obj.feature_names_in_)
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            names = feature_names(step)
            if names is not None:
                return names
    if getattr(obj, "estimators_", None):
        return feature_names(obj.estimators_[0])
    if getattr(obj, "steps", None):
        return feature_names(obj.steps[0][1])
    return None

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

candidates = [
    ("baseline", inputs["07_model_train_baseline_2N13J4R1/model.pkl"]),
    ("alternative", inputs["08_model_train_alternative_JE7KKHHV/alternative_model.pkl"]),
]

y_true = val[targets].to_numpy()
rows = []
results = []

for name, path in candidates:
    model = load_pkl(path)
    fnames = feature_names(model)
    if fnames is None:
        raise ValueError(f"No feature_names_in_ found for candidate {name}")
    missing = [c for c in fnames if c not in val.columns]
    if missing:
        raise KeyError(f"Missing feature columns for {name}: {missing}")
    X = val.loc[:, fnames]
    preds = np.asarray(model.predict(X))
    if preds.ndim == 1:
        preds = preds.reshape(-1, len(targets))

    f1s, hs = [], []
    for j, target in enumerate(targets):
        yt = y_true[:, j]
        yp = preds[:, j]
        f1s.append(float(f1_score(yt, yp, average="macro", zero_division=0)))
        hs.append(float(hamming_loss(yt, yp)))

    macro = float(np.mean(f1s))
    ham = float(np.mean(hs))
    subset = float(np.mean(np.all(preds == y_true, axis=1)))
    results.append({"model": name, "macro_f1": macro, "hamming_loss": ham, "subset_accuracy": subset})

    for metric, value in [
        ("macro_f1", macro),
        ("hamming_loss", ham),
        ("subset_accuracy", subset),
    ]:
        rows.append({"model": name, "metric": metric, "value": float(f"{value:.4g}")})
    for j, target in enumerate(targets):
        rows.append({"model": name, "metric": f"macro_f1_{target}", "value": float(f"{f1s[j]:.4g}")})
        rows.append({"model": name, "metric": f"hamming_loss_{target}", "value": float(f"{hs[j]:.4g}")})

best = max(results, key=lambda r: (r["macro_f1"], -r["hamming_loss"]))
best_name = best["model"]
best_score = best["macro_f1"]
best_score_rounded = float(f"{best_score:.4g}")

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics for multilabel candidates")

mlca_runtime.record_metric(primary_metric, best_score, "val")
mlca_runtime.record_metric("hamming_loss", best["hamming_loss"], "val")
mlca_runtime.record_metric("subset_accuracy", best["subset_accuracy"], "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", best_score_rounded, source="validation_metrics.csv")

print(len(results), "candidates scored")
print(best_name, best_score_rounded)
print(metrics_df.shape)
