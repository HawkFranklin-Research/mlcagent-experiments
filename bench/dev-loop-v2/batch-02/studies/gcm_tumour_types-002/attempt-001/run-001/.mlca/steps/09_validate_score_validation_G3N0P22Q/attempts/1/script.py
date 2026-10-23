import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing protocol key {k!r}; available keys: {list(d)}")
    return d[k]

targets = req(proto, "targets")
excluded = req(proto, "excluded_columns")
task = req(proto, "task")
primary = req(task, "primary_metric")
group = proto.get("group_column")
target = targets[0]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

pre = pickle.loads(inp["05_preprocess_fit_preprocessing_0GNR753P/preprocessor.pkl"].read_bytes())
sel = pickle.loads(inp["06_features_select_features_MJW9JFWH/selector.pkl"].read_bytes())

excl = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = []
for c in train.columns:
    if c in excl:
        continue
    s = train[c]
    if s.dtype.kind in "OUS" and s.notna().all() and s.nunique(dropna=False) == len(train):
        continue
    if s.dtype.kind in "iu" and s.nunique(dropna=False) == len(train) and any(k in c.lower() for k in ("id", "code", "index")):
        continue
    feature_cols.append(c)

y_val = val[target]
PRE_KEYS = ("pre", "scaler", "imput", "encode", "vector")
SEL_KEYS = ("select", "feature")

def has_step(obj, keys):
    return isinstance(obj, Pipeline) and any(any(k in n.lower() for k in keys) for n in obj.named_steps)

def input_cols(obj):
    try:
        if hasattr(obj, "feature_names_in_"):
            return list(obj.feature_names_in_)
    except Exception:
        pass
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            cols = input_cols(step)
            if cols is not None:
                return cols
    return None

def accepts_raw(obj):
    cols = input_cols(obj)
    return cols is not None and all(c in feature_cols for c in cols)

def make_pipe(model):
    if has_step(model, PRE_KEYS):
        return model
    if has_step(model, SEL_KEYS):
        return Pipeline([("preprocessor", pre), ("model", model)])
    if accepts_raw(model):
        return model
    if has_step(sel, PRE_KEYS):
        return Pipeline([("selector", sel), ("model", model)])
    return Pipeline([("preprocessor", pre), ("selector", sel), ("model", model)])

def get_classes(pipe):
    try:
        return list(pipe.classes_)
    except Exception:
        est = pipe
        while isinstance(est, Pipeline):
            est = est.steps[-1][1]
        return list(est.classes_)

def score(pipe):
    cols = input_cols(pipe) or feature_cols
    missing = [c for c in cols if c not in feature_cols]
    if missing:
        raise ValueError(f"Pipeline expects non-feature columns: {missing}")
    X = val[cols]
    proba = pipe.predict_proba(X)
    classes = np.asarray(get_classes(pipe))
    present = [c for c in classes if (y_val == c).any()]
    if len(present) == len(classes):
        auc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)
    else:
        aucs = []
        for c in present:
            j = int(np.where(classes == c)[0][0])
            aucs.append(roc_auc_score((y_val == c).astype(int), proba[:, j]))
        auc = float(np.mean(aucs))
    pred = pipe.predict(X)
    f1 = f1_score(y_val, pred, average="macro", labels=classes)
    return float(auc), float(f1)

def manifest_name(path, fallback):
    try:
        m = json.loads(path.read_text())
        for k in ("model_name", "name", "estimator_name", "model"):
            if isinstance(m.get(k), str):
                return m[k]
    except Exception:
        pass
    return fallback

candidates = [
    ("logistic", inp["07_model_train_logistic_N3PQQ99D/model.pkl"],
     inp["07_model_train_logistic_N3PQQ99D/manifest.json"]),
    ("alternative", inp["08_model_train_alternative_SGYXE5GB/alternative_model.pkl"],
     inp["08_model_train_alternative_SGYXE5GB/manifest.json"]),
]

rows = []
results = []
for fallback, mpath, manpath in candidates:
    name = manifest_name(manpath, fallback)
    model = pickle.loads(mpath.read_bytes())
    pipe = make_pipe(model)
    auc, f1 = score(pipe)
    results.append((name, auc, f1))
    rows.append({"model": name, "metric": primary, "value": auc, "threshold": np.nan})
    rows.append({"model": name, "metric": "f1_macro", "value": f1, "threshold": np.nan})
    mlca_runtime.record_metric(primary, auc, split="val")

table = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
table["value"] = table["value"].map(lambda v: float(f"{v:.4g}"))
mlca_runtime.save_table(table, "validation_metrics.csv", caption="Validation metrics by candidate model")

best_name, best_auc, _ = max(results, key=lambda r: r[1])
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")

print(f"validation rows={len(val)}, features={len(feature_cols)}")
print(f"best_model={best_name}")
print(f"{primary} val={best_auc:.4f}")
