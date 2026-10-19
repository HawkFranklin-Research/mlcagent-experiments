import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for k in ["targets", "excluded_columns", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing {k}; available top-level keys: {list(protocol.keys())}")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; task keys: {list(task.keys())}")
primary = task["primary_metric"]
group = protocol.get("group_column")
group_cols = [group] if group else []

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
forbidden = set(targets) | set(excluded) | set(group_cols)
feature_cols = []
for c in train.columns:
    if c in forbidden:
        continue
    s = train[c]
    all_distinct = s.nunique(dropna=True) == len(s)
    key_like = all_distinct and (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_integer_dtype(s)
    )
    if not key_like:
        feature_cols.append(c)

def to_pred_frame(pred, index):
    if isinstance(pred, pd.DataFrame):
        if all(t in pred.columns for t in targets):
            return pred[targets].reset_index(drop=True)
        pred = pred.to_numpy()
    if isinstance(pred, (list, tuple)):
        pred = np.column_stack([np.asarray(a) for a in pred])
    pred = np.asarray(pred)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if pred.shape[1] != len(targets):
        raise ValueError(f"prediction shape {pred.shape} does not match targets {targets}")
    return pd.DataFrame(pred, columns=targets)

def align_labels(train_s, true_s, pred_s):
    train_s = pd.Series(train_s).dropna()
    true_s = pd.Series(true_s)
    pred_s = pd.Series(pred_s)
    if pd.api.types.is_numeric_dtype(pred_s) and not pd.api.types.is_numeric_dtype(true_s):
        labels = sorted(train_s.unique(), key=lambda x: str(x))
        def decode(v):
            if pd.isna(v):
                return v
            try:
                return labels[int(round(float(v)))]
            except Exception:
                return v
        pred_s = pred_s.map(decode)
    if pd.api.types.is_numeric_dtype(true_s) != pd.api.types.is_numeric_dtype(pred_s):
        true_s = true_s.astype(str)
        pred_s = pred_s.astype(str)
    if not pd.api.types.is_numeric_dtype(true_s) and not pd.api.types.is_numeric_dtype(pred_s):
        true_s = true_s.astype(str)
        pred_s = pred_s.astype(str)
    return true_s, pred_s

def input_for_model(model):
    expected = None
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    elif hasattr(model, "steps"):
        first = model.steps[0][1]
        if hasattr(first, "feature_names_in_"):
            expected = list(first.feature_names_in_)
    if expected:
        bad = [c for c in expected if c in forbidden]
        if bad:
            raise ValueError(f"model expects forbidden columns: {bad}")
        missing = [c for c in expected if c not in val.columns]
        if missing:
            raise KeyError(f"validation missing model input columns: {missing}")
        return val.loc[:, expected].copy()
    return val.loc[:, feature_cols].copy()

def score_model(model, name):
    pred = to_pred_frame(model.predict(input_for_model(model)), val.index)
    f1s, hls, matches = [], [], []
    rows = []
    for t in targets:
        yt = val[t].reset_index(drop=True)
        pt = pred[t].reset_index(drop=True)
        yt, pt = align_labels(train[t].reset_index(drop=True), yt, pt)
        f1 = float(f1_score(yt, pt, average="macro", zero_division=0))
        hl = float(hamming_loss(yt, pt))
        f1s.append(f1)
        hls.append(hl)
        matches.append(yt.astype(str).to_numpy() == pt.astype(str).to_numpy())
        rows.append((name, f"macro_f1_{t}", f1))
        rows.append((name, f"hamming_loss_{t}", hl))
    macro_f1 = float(np.mean(f1s))
    hamming_avg = float(np.mean(hls))
    subset_acc = float(np.mean(np.logical_and.reduce(matches)))
    rows += [
        (name, "macro_f1", macro_f1),
        (name, "hamming_loss", hamming_avg),
        (name, "subset_accuracy", subset_acc),
    ]
    return macro_f1, rows

candidates = [
    ("baseline", inputs["07_model_train_baseline_8KXAGBNQ/model.pkl"]),
    ("alternative", inputs["08_model_train_alternative_K6DZGSFY/alternative_model.pkl"]),
]
metric_rows = []
scores = {}
for name, path in candidates:
    with open(path, "rb") as f:
        model = pickle.load(f)
    score, rows = score_model(model, name)
    scores[name] = score
    metric_rows.extend(rows)

metrics = pd.DataFrame(metric_rows, columns=["model", "metric", "value"])
metrics["value"] = metrics["value"].map(lambda v: float(f"{v:.4g}"))
best_model = max(scores, key=scores.get)
best_value = scores[best_model]
mlca_runtime.save_table(metrics, "validation_metrics.csv", caption="Validation metrics by candidate model")
mlca_runtime.record_metric(primary, best_value, "val")
mlca_runtime.record_fact_candidate("best_validation_score", best_value, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
print(f"scored {len(scores)} candidates on val")
print(f"best_model={best_model} {primary}={best_value:.4f}")
print(f"saved validation_metrics.csv rows={len(metrics)}")
