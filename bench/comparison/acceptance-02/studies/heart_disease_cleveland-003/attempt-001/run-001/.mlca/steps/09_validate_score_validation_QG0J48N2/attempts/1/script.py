import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

proto = json.loads(inp["_protocol"].read_text())
targets = proto["targets"]
excluded = list(proto.get("excluded_columns", []))
group_col = proto.get("group_column")
primary = proto["task"]["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
feature_cols = []
for c in cols:
    s = train[c]
    if s.notna().all() and s.is_unique and (pd.api.types.is_integer_dtype(s) or not pd.api.types.is_numeric_dtype(s)):
        continue
    feature_cols.append(c)

X_val = val[feature_cols].copy()
Y = {t: val[t].to_numpy() for t in targets}

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load(inp["05_preprocess_fit_preprocessing_DXHK0F72/preprocessor.pkl"])
sel = load(inp["06_features_retain_features_NH52E6T3/selector_all.pkl"])
m_base = load(inp["07_model_train_baseline_X4GJNHE4/model.pkl"])
m_alt = load(inp["08_model_train_alternative_5NAZBJJ0/alternative_model.pkl"])

def fnames(obj):
    try:
        return list(obj.feature_names_in_)
    except Exception:
        return None

def frontend(pre, sel, raw):
    si = fnames(sel)
    if si is not None and set(si) == set(raw):
        return Pipeline([("sel", sel)])
    return Pipeline([("pre", pre), ("sel", sel)])

def full(model, pre, sel, raw):
    mi = fnames(model)
    if isinstance(model, Pipeline) and mi is not None and set(mi) == set(raw):
        return model
    fe = frontend(pre, sel, raw)
    steps = list(fe.steps)
    if isinstance(model, Pipeline):
        steps += list(model.steps)
    else:
        steps += [("model", model)]
    return Pipeline(steps)

def evaluate(name, model):
    preds = np.asarray(model.predict(X_val))
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    rows, f1s, hls = [], [], []
    for i, t in enumerate(targets):
        yt = Y[t]
        yp = preds[:, i]
        if t == targets[0] and pd.api.types.is_integer_dtype(yt) and not pd.api.types.is_integer_dtype(yp):
            yp = yp.astype(yt.dtype)
        f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
        hl = float(np.mean(np.asarray(yt) != np.asarray(yp)))
        rows.append((name, f"f1_{t}", round(f1, 4)))
        rows.append((name, f"hamming_loss_{t}", round(hl, 4)))
        f1s.append(f1); hls.append(hl)
    mf1 = float(np.mean(f1s))
    mhl = float(np.mean(hls))
    Ymat = np.stack([Y[t] for t in targets], axis=1)
    sub = float(np.mean(np.all(Ymat == preds, axis=1)))
    rows += [
        (name, "macro_f1", round(mf1, 4)),
        (name, "hamming_loss", round(mhl, 4)),
        (name, "subset_accuracy", round(sub, 4)),
    ]
    return mf1, rows

cands = [("baseline", full(m_base, pre, sel, feature_cols)),
         ("alternative", full(m_alt, pre, sel, feature_cols))]
all_rows, scores = [], {}
for name, mdl in cands:
    s, rows = evaluate(name, mdl)
    scores[name] = s
    all_rows += rows

df = pd.DataFrame(all_rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(df, "validation_metrics.csv", "Validation metrics per candidate")

best = max(scores, key=scores.get)
best_score = float(scores[best])
mlca_runtime.record_metric(primary, best_score, "val")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best, source="validation_metrics.csv")

print(f"candidates={len(scores)} features={len(feature_cols)}")
print(f"best={best} {primary}={best_score:.4f}")
