import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime as mlca
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

mlca.set_seed(42)
inp = mlca.inputs()
proto = json.loads(inp["_protocol"].read_text())
for k in ("task", "targets", "excluded_columns"):
    if k not in proto:
        raise KeyError(f"Missing protocol key {k!r}. Available: {list(proto)}")
task = proto["task"]
primary = task["primary_metric"]
targets = list(proto["targets"])
excluded = set(proto["excluded_columns"])
group = proto.get("group_column")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
for t in targets:
    if t not in train.columns or t not in val.columns:
        raise KeyError(f"Missing target column {t!r}")

feature_cols = []
for c in train.columns:
    if c in excluded or c in targets or (group is not None and c == group):
        continue
    s = train[c]
    all_distinct = s.nunique(dropna=True) == len(train)
    key_like = (pd.api.types.is_integer_dtype(s) or pd.api.types.is_object_dtype(s) or
                pd.api.types.is_string_dtype(s))
    if all_distinct and key_like:
        continue
    feature_cols.append(c)
if not feature_cols:
    raise ValueError("No feature columns remain after exclusions")

def load_pickle(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

def predict_obj(obj):
    names = getattr(obj, "feature_names_in_", None)
    if names is not None and all(n in val.columns for n in names):
        try:
            return obj.predict(val[list(names)])
        except Exception:
            if isinstance(obj, Pipeline):
                raise
    if isinstance(obj, Pipeline):
        return obj.predict(val[feature_cols])
    pre = load_pickle("05_preprocess_fit_preprocessing_7SHSFSVP/preprocessor.pkl")
    sel = load_pickle("06_features_retain_features_JQTP2HCD/selector_all.pkl")
    if hasattr(pre, "set_output"):
        pre.set_output(transform="pandas")
    steps = []
    if isinstance(sel, Pipeline) and any(step is pre for step in sel.named_steps.values()):
        steps.append(("front", sel))
    elif getattr(sel, "feature_names_in_", None) is not None and all(
        n in val.columns for n in sel.feature_names_in_
    ):
        steps.append(("front", sel))
    else:
        steps.append(("front", Pipeline([("pre", pre), ("sel", sel)])))
    steps.append(("model", obj))
    return Pipeline(steps).predict(val[feature_cols])

candidates = {
    "baseline": load_pickle("07_model_train_baseline_VKCYA88G/model.pkl"),
    "alternative": load_pickle("08_model_train_alternative_65RN6RC0/alternative_model.pkl"),
}
if primary not in ("rmse", "mae", "r2"):
    raise ValueError(f"Unsupported primary metric {primary!r}")

rows, macro_scores = [], {}
for name, obj in candidates.items():
    pred = np.asarray(predict_obj(obj))
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    per_target = {}
    for i, target in enumerate(targets):
        yt, yp = val[target].to_numpy(), pred[:, i]
        per_target[(target, "rmse")] = float(np.sqrt(mean_squared_error(yt, yp)))
        per_target[(target, "mae")] = float(mean_absolute_error(yt, yp))
        per_target[(target, "r2")] = float(r2_score(yt, yp))
        for m in ("rmse", "mae", "r2"):
            rows.append({"model": name, "metric": f"{target}_{m}", "value": per_target[(target, m)]})
    for m in ("rmse", "mae", "r2"):
        macro = float(np.mean([per_target[(t, m)] for t in targets]))
        rows.append({"model": name, "metric": f"macro_{m}", "value": macro})
        if m == primary:
            rows.append({"model": name, "metric": primary, "value": macro})
    macro_scores[name] = float(np.mean([per_target[(t, primary)] for t in targets]))

lower_better = primary in ("rmse", "mae", "mse", "loss", "error") or "error" in primary
best_name = min(macro_scores, key=macro_scores.get) if lower_better else max(macro_scores, key=macro_scores.get)
best_score = macro_scores[best_name]

out = pd.DataFrame(rows)
out["value"] = out["value"].map(lambda v: float(f"{v:.4g}"))
mlca.save_table(out, "validation_metrics.csv", caption="Validation metrics by candidate")
mlca.record_metric(primary, best_score, split="val")
mlca.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")
print(f"val rows={len(val)} features={len(feature_cols)} targets={targets}")
print(f"best={best_name} {primary}={best_score:.4g}")
