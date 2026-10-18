import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

def need(d, k, label="protocol"):
    if k not in d:
        raise KeyError(f"Missing {label} key {k!r}; available: {sorted(d)}")
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
    if len(s) == 0 or s.nunique(dropna=False) != len(s):
        return False
    if isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
        return True
    return pd.api.types.is_integer_dtype(s)

drop = set(targets) | set(excluded) | {group}
features = [c for c in train.columns if c not in drop and not is_key_like(train[c])]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

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
    names = getattr(pipe, "feature_names_in_", None)
    if names is not None:
        return list(names)
    if isinstance(pipe, Pipeline):
        for _, step in pipe.steps:
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                return list(names)
    return None

def classes_of(pipe):
    if isinstance(pipe, Pipeline):
        return list(getattr(pipe.steps[-1][1], "classes_"))
    return list(getattr(pipe, "classes_"))

model_specs = [
    ("08_logistic", inp["08_model_train_logistic_GP5KS1CK/model.pkl"]),
    ("09_alternative", inp["09_model_train_alternative_16WYVMV8/alternative_model.pkl"]),
]
latest = inp.get("latest_model")
if latest is not None and latest.resolve() not in {p.resolve() for _, p in model_specs}:
    model_specs.append(("latest_model", latest))

y_val = val[target].to_numpy()
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
    pred = pipe.predict(X_val)

    present = [c for c in classes if c in set(y_val)]
    if len(present) == len(classes):
        auc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)
    else:
        idx = [classes.index(c) for c in present]
        auc = roc_auc_score(y_val, proba[:, idx], multi_class="ovr", average="macro", labels=present)

    f1 = f1_score(y_val, pred, average="macro", labels=classes, zero_division=0)
    rows.append({"model": name, "metric": primary, "value": float(f"{auc:.4g}"), "threshold": np.nan})
    rows.append({"model": name, "metric": "macro_f1", "value": float(f"{f1:.4g}"), "threshold": np.nan})
    results.append({"model": name, "primary": float(auc), "macro_f1": float(f1)})

metrics = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
mlca_runtime.save_table(metrics, "validation_metrics.csv", caption="Validation metrics by candidate model")

best = max(results, key=lambda r: r["primary"])
for r in sorted(results, key=lambda r: r["primary"]):
    mlca_runtime.record_metric(primary, r["primary"], "val")
mlca_runtime.record_fact_candidate("best_model", best["model"], source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best["primary"], source="validation_metrics.csv")

print(f"train={train.shape} val={val.shape} features={len(features)} candidates={len(results)}")
print(f"primary={primary} best={best['model']} val={best['primary']:.4g}")
