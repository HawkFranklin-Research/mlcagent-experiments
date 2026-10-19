import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, hamming_loss
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
if "group_column" not in proto:
    raise KeyError(f"Missing group_column; top-level keys: {list(proto.keys())}")
targets = proto["targets"]
excluded = proto["excluded_columns"]
group_col = proto["group_column"]
primary = proto["task"]["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
exclude = set(targets) | set(excluded)
if group_col is not None:
    exclude.add(group_col)

def key_like(c):
    s = train[c]
    if s.isna().any():
        return False
    return (s.dtype.kind in "OUS" and s.nunique() == len(train)) or (s.dtype.kind in "iu" and s.nunique() == len(train))

feature_cols = [c for c in train.columns if c not in exclude and not key_like(c)]

def load_pkl(k):
    with open(inputs[k], "rb") as f:
        return pickle.load(f)

pre = load_pkl("05_preprocess_fit_preprocessing_WHDXCN7B/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_J1HGBJXK/selector_all.pkl")

def has_pre(obj):
    return isinstance(obj, Pipeline) and any(n in obj.named_steps for n in ["preprocessor", "preprocess"])

def make_pipe(model):
    fin = getattr(model, "feature_names_in_", None)
    if isinstance(model, Pipeline):
        if has_pre(model):
            return model
        if fin is not None and set(fin).issubset(set(feature_cols)):
            return model
    elif fin is not None and set(fin).issubset(set(feature_cols)):
        return model
    if has_pre(sel):
        return Pipeline([("selector", sel), ("model", model)])
    steps = []
    if pre is not None:
        steps.append(("preprocessor", pre))
    if sel is not None:
        steps.append(("selector", sel))
    steps.append(("model", model))
    return Pipeline(steps)

def align_pred(pred, truth, train_truth):
    pred = np.asarray(pred)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    truth = np.asarray(truth)
    if truth.ndim == 1:
        truth = truth.reshape(-1, 1)
    out = pred.astype(object) if pred.dtype.kind in "iu" else pred.copy()
    for j in range(truth.shape[1]):
        if pred[:, j].dtype.kind in "iu" and truth[:, j].dtype.kind in "OUS":
            uniq = np.sort(pd.unique(train_truth.iloc[:, j].dropna()))
            mapping = {i: uniq[i] for i in range(len(uniq))}
            out[:, j] = [mapping.get(int(x), x) for x in pred[:, j]]
    return out

def score(y_true, y_pred):
    f1s, hls = [], []
    for j in range(y_true.shape[1]):
        f1s.append(f1_score(y_true[:, j], y_pred[:, j], average="macro", zero_division=0))
        hls.append(hamming_loss(y_true[:, j], y_pred[:, j]))
    subset = float(np.mean(np.all(y_true == y_pred, axis=1)))
    return float(np.mean(f1s)), float(np.mean(hls)), subset, f1s, hls

cands = {
    "baseline": "07_model_train_baseline_R0Q02H91/model.pkl",
    "alternative": "08_model_train_alternative_025XEA6J/alternative_model.pkl",
}
y_true = val[targets].to_numpy()
results = {}
for name, key in cands.items():
    model = load_pkl(key)
    pipe = make_pipe(model)
    front = pipe.steps[0][1] if isinstance(pipe, Pipeline) else pipe
    exp = getattr(front, "feature_names_in_", None)
    if exp is not None:
        bad = set(exp) & exclude
        if bad:
            raise ValueError(f"Target/excluded column reached model: {bad}")
        cols = list(exp)
    else:
        cols = [c for c in feature_cols if c in val.columns]
    pred = pipe.predict(val.loc[:, cols].copy())
    pred = align_pred(pred, y_true, train[targets])
    results[name] = score(y_true, pred)

rows = []
for name, (mf1, hl, sub, f1s, hls) in results.items():
    rows.append({"model": name, "metric": primary, "value": mf1})
    rows.append({"model": name, "metric": "hamming_loss", "value": hl})
    rows.append({"model": name, "metric": "subset_accuracy", "value": sub})
    for j, t in enumerate(targets):
        rows.append({"model": name, "metric": f"macro_f1_{t}", "value": f1s[j]})
        rows.append({"model": name, "metric": f"hamming_loss_{t}", "value": hls[j]})
df = pd.DataFrame(rows)
df["value"] = df["value"].map(lambda x: float(f"{x:.4g}"))
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation metrics")

lower_better = any(k in primary.lower() for k in ["loss", "error", "rmse", "mae"])
best_name = min(results, key=lambda n: results[n][0]) if lower_better else max(results, key=lambda n: results[n][0])
best_score = results[best_name][0]
mlca_runtime.record_metric(primary, float(best_score), split="val")
mlca_runtime.record_fact_candidate("best_validation_score", float(best_score), source="validation_metrics.csv")
print(f"evaluated {len(results)} models on {len(val)} val rows; best {best_name} {primary}={best_score:.4g}")
