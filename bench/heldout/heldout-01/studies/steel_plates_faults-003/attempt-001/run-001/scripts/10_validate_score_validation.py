import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

if "targets" not in protocol:
    raise KeyError(f"Missing 'targets'. Available top-level keys: {list(protocol)}")
if "excluded_columns" not in protocol:
    raise KeyError(f"Missing 'excluded_columns'. Available top-level keys: {list(protocol)}")
if "task" not in protocol or "primary_metric" not in protocol["task"]:
    raise KeyError(f"Missing task.primary_metric. Available top-level keys: {list(protocol)}")

targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group_col = protocol.get("group_column")
primary = protocol["task"]["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

features = [c for c in train.columns if c not in excluded and c not in targets and c != group_col]
for c in list(features):
    s = train[c]
    if s.nunique(dropna=False) == len(s) and (
        pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)
    ):
        features.remove(c)

with open(inputs["06_preprocess_fit_preprocessing_HH50E3RG/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["07_features_retain_features_Y0FEJMPH/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)

candidates = [
    ("baseline", inputs["08_model_train_baseline_6RRDC1E6/model.pkl"]),
    ("alternative", inputs["09_model_train_alternative_D4ZR7T7K/alternative_model.pkl"]),
]

def predict_saved(model, frame, raw_features, pre, sel):
    cols = list(getattr(model, "feature_names_in_", []))
    if cols and set(cols).issubset(frame.columns):
        return model.predict(frame[cols])
    scols = list(getattr(sel, "feature_names_in_", [])) if sel is not None else []
    if scols and set(scols).issubset(frame.columns):
        return model.predict(sel.transform(frame[scols]))
    X = pre.transform(frame[raw_features]) if pre is not None else frame[raw_features]
    X = sel.transform(X) if sel is not None else X
    return model.predict(X)

def score_predictions(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    f1s, hams = [], []
    for i, _ in enumerate(targets):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        f1s.append(float(f1_score(yt, yp, average="macro", zero_division=0)))
        hams.append(float(np.mean(yt != yp)))
    subset = float(np.mean(np.all(y_pred == y_true, axis=1)))
    return f1s, float(np.mean(f1s)), float(np.mean(hams)), subset

rows = []
primary_scores = []
for name, path in candidates:
    with open(path, "rb") as f:
        model = pickle.load(f)
    pred = predict_saved(model, val, features, preprocessor, selector)
    f1s, macro_f1, hamming_avg, subset_acc = score_predictions(val[targets].to_numpy(), pred)
    primary_scores.append(macro_f1)
    rows.append({"model": name, "metric": primary, "value": round(macro_f1, 4)})
    for t, v in zip(targets, f1s):
        rows.append({"model": name, "metric": f"f1_{t}", "value": round(v, 4)})
    rows.append({"model": name, "metric": "hamming_loss", "value": round(hamming_avg, 4)})
    rows.append({"model": name, "metric": "subset_accuracy", "value": round(subset_acc, 4)})

metrics = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(metrics, "validation_metrics.csv", caption="Validation metrics by candidate model")

best_i = int(np.argmax(primary_scores))
best_name = candidates[best_i][0]
best_score = float(primary_scores[best_i])

mlca_runtime.record_metric(primary, best_score, split="val")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")

print(f"train_shape={train.shape} val_shape={val.shape} targets={len(targets)}")
print(f"features={len(features)} candidates={len(candidates)}")
print(f"best_model={best_name} {primary}={best_score:.4f}")
