import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

def need(d, k, label="protocol"):
    if k not in d:
        raise KeyError(f"Missing {label!r} key {k!r}; available: {sorted(d)}")
    return d[k]

task = need(protocol, "task")
primary = need(task, "primary_metric", "task")
targets = list(need(protocol, "targets"))
excluded = list(need(protocol, "excluded_columns"))
group = need(protocol, "group_column")
target = targets[0]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def is_key_like(s):
    if len(s) == 0 or s.nunique(dropna=False) != len(s): return False
    if isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s): return True
    return pd.api.types.is_integer_dtype(s)

drop = set(targets) | set(excluded) | {group}
features = [c for c in train.columns if c not in drop and not is_key_like(train[c])]

def load_pkl(path):
    with open(path, "rb") as f: return pickle.load(f)

pre = load_pkl(inp["06_preprocess_fit_preprocessing_MYPQXFW6/preprocessor.pkl"])
sel = load_pkl(inp["07_features_select_features_NK102NGP/selector.pkl"])

def with_frontend(model):
    if isinstance(model, Pipeline) and len(model.named_steps) > 1:
        return model
    est = model.named_steps[list(model.named_steps)[0]] if isinstance(model, Pipeline) else model
    if isinstance(sel, Pipeline) and any("pre" in n.lower() or "column" in n.lower() for n in sel.named_steps):
        return Pipeline([("selector", sel), ("model", est)])
    return Pipeline([("preprocessor", pre), ("selector", sel), ("model", est)])

def input_names(pipe):
    n = getattr(pipe, "feature_names_in_", None)
    if n is not None: return list(n)
    if isinstance(pipe, Pipeline):
        for _, step in pipe.steps:
            n = getattr(step, "feature_names_in_", None)
            if n is not None: return list(n)
    return None

def classes_of(pipe):
    if isinstance(pipe, Pipeline): return list(pipe.steps[-1][1].classes_)
    return list(getattr(pipe, "classes_"))

model_specs = [
    ("08_logistic", inp["08_model_train_logistic_GP5KS1CK/model.pkl"]),
    ("09_alternative", inp["09_model_train_alternative_16WYVMV8/alternative_model.pkl"]),
]
latest = inp.get("latest_model")
if latest is not None and latest.resolve() not in {p.resolve() for _, p in model_specs}:
    model_specs.append(("latest_model", latest))

y_val = np.array([str(v) for v in val[target].to_numpy()])
rows, results = [], []
for name, path in model_specs:
    pipe = with_frontend(load_pkl(path))
    names = input_names(pipe)
    use_cols = names if names is not None else features
    bad = set(use_cols) & (set(targets) | set(excluded) | {group})
    if bad:
        raise ValueError(f"Model input contains forbidden columns: {sorted(bad)}")
    missing = [c for c in use_cols if c not in val.columns]
    if missing:
        raise KeyError(f"Validation missing model input columns: {missing}")
    X_val = val.loc[:, use_cols]
    classes = classes_of(pipe)
    proba = pipe.predict_proba(X_val)
    pred = np.array([str(v) for v in pipe.predict(X_val)])
    cls_str = [str(c) for c in classes]
    idx_map = {c: i for i, c in enumerate(cls_str)}
    mask = np.array([v in idx_map for v in y_val])
    y_al = y_val[mask]
    p_al = proba[mask]
    pred_al = pred[mask]
    aucs, ok_cls = [], []
    for j, c in enumerate(cls_str):
        y_bin = (y_al == c).astype(int)
        if y_bin.min() == y_bin.max(): continue
        aucs.append(roc_auc_score(y_bin, p_al[:, j]))
        ok_cls.append(c)
    auc = float(np.mean(aucs)) if aucs else float("nan")
    f1 = f1_score(y_al, pred_al, labels=cls_str, average="macro", zero_division=0)
    rows.append({"model": name, "metric": primary, "value": float(f"{auc:.4g}") if not np.isnan(auc) else np.nan, "threshold": np.nan})
    rows.append({"model": name, "metric": "macro_f1", "value": float(f"{float(f1):.4g}"), "threshold": np.nan})
    results.append({"model": name, "primary": auc, "macro_f1": float(f1)})

metrics = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
mlca_runtime.save_table(metrics, "validation_metrics.csv", caption="Validation metrics by candidate model")

valid = [r for r in results if not np.isnan(r["primary"])]
best = max(valid, key=lambda r: r["primary"]) if valid else results[0]
for r in results:
    if not np.isnan(r["primary"]):
        mlca_runtime.record_metric(primary, r["primary"], "val")
mlca_runtime.record_fact_candidate("best_model", best["model"], source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best["primary"]), source="validation_metrics.csv")

print(f"train={train.shape} val={val.shape} features={len(features)} candidates={len(results)}")
print(f"primary={primary} best={best['model']} val={best['primary']:.4g}")
