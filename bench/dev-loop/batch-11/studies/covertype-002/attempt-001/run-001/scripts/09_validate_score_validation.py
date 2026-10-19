import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
try:
    task = protocol["task"]
    targets = list(protocol["targets"])
    primary = task["primary_metric"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; top-level keys are {list(protocol.keys())}") from e

excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
excl = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in excl]
feature_cols = [
    c for c in feature_cols
    if not (
        (pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]))
        and train[c].nunique(dropna=False) == len(train)
    )
]
Xval = val[feature_cols]
y_val = val[targets[0]]

def load(k):
    return pickle.loads(inp[k].read_bytes())

pre = load("05_preprocess_fit_preprocessing_8APC2384/preprocessor.pkl")
sel = load("06_features_select_features_23ZRTGDS/selector.pkl")
models = {
    "logistic": load("07_model_train_logistic_H5RFG7PA/model.pkl"),
    "alternative": load("08_model_train_alternative_5HD55PJ2/alternative_model.pkl"),
}

def cols_of(obj, default):
    if obj is None:
        return default
    c = getattr(obj, "feature_names_in_", None)
    if c is not None:
        return list(c)
    if isinstance(obj, Pipeline):
        for st in obj.named_steps.values():
            c = getattr(st, "feature_names_in_", None)
            if c is not None:
                return list(c)
    return default

pre_has_sel = isinstance(pre, Pipeline) and any("select" in k.lower() for k in pre.named_steps)

def trans_bare(X):
    if sel is not None:
        sc = cols_of(sel, None)
        if sc is not None and set(sc).issubset(X.columns):
            return sel.transform(X[sc])
    if pre is not None:
        pc = cols_of(pre, feature_cols)
        if not set(pc).issubset(X.columns):
            pc = feature_cols
        Xt = pre.transform(X[pc])
        if sel is not None and not pre_has_sel:
            Xt = sel.transform(Xt)
        return Xt
    if sel is not None:
        return sel.transform(X[cols_of(sel, feature_cols)])
    return X

def proba_for(model, X):
    if isinstance(model, Pipeline):
        mc = cols_of(model, feature_cols)
        if set(mc).issubset(X.columns):
            return model.predict_proba(X[mc]), model
        return model.predict_proba(trans_bare(X)), model
    mc = getattr(model, "feature_names_in_", None)
    if mc is not None:
        mc = list(mc)
        if set(mc) == set(feature_cols) and set(mc).issubset(X.columns):
            return model.predict_proba(X[mc]), model
    Xt = trans_bare(X)
    if isinstance(Xt, pd.DataFrame) and mc is not None:
        if set(mc).issubset(Xt.columns):
            Xt = Xt[mc]
        elif len(mc) == Xt.shape[1]:
            Xt = Xt.copy()
            Xt.columns = mc
    return model.predict_proba(Xt), model

def classes_of(model):
    c = getattr(model, "classes_", None)
    if c is not None:
        return np.asarray(c)
    if isinstance(model, Pipeline):
        return np.asarray(model.named_steps[list(model.named_steps)[-1]].classes_)
    raise AttributeError("model has no classes_")

def sig4(x):
    return float(f"{float(x):.4g}")

rows, scores = [], {}
for name, model in models.items():
    proba, _ = proba_for(model, Xval)
    proba = np.asarray(proba)
    classes = classes_of(model)
    present = [c for c in classes if np.any(y_val == c)]
    if len(present) == len(classes):
        auc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)
    else:
        auc = np.mean([
            roc_auc_score((y_val == c).astype(int), proba[:, j])
            for j, c in enumerate(classes) if np.any(y_val == c)
        ])
        print(f"classes missing for {name}: {sorted(set(classes) - set(present))}")
    pred = classes[np.argmax(proba, axis=1)]
    f1 = f1_score(y_val, pred, labels=classes, average="macro")
    scores[name] = auc
    rows.append({"model": name, "metric": primary, "value": sig4(auc), "threshold": np.nan})
    rows.append({"model": name, "metric": "macro_f1", "value": sig4(f1), "threshold": np.nan})
    mlca_runtime.record_metric(primary, float(auc), split="val")

lower = any(t in primary.lower() for t in ("loss", "error", "brier", "mae", "mse", "rmse", "logloss"))
best = min(scores, key=scores.get) if lower else max(scores, key=scores.get)
table = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
mlca_runtime.save_table(table, "validation_metrics.csv", caption="Validation metrics for candidate models.")
mlca_runtime.record_fact_candidate("best_model", best, source="validation_metrics.csv")
if primary == "roc_auc_ovr":
    mlca_runtime.record_fact_candidate("best_val_roc_auc", sig4(scores[best]), source="validation_metrics.csv")
print(f"val rows={len(val)} features={len(feature_cols)}")
print(f"models={len(models)} primary={primary}")
print(f"best={best} value={scores[best]:.4g}")
