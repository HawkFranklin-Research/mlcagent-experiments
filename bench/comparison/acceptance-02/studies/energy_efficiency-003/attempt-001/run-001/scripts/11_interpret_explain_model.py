import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())

for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in proto:
        raise KeyError(f"Missing protocol key {k!r}; available keys: {sorted(proto)}")

targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group_col = proto["group_column"]
task = proto["task"]
primary_metric = task["primary_metric"]

train = pd.read_parquet(paths["train"])
feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
for c in list(feature_cols):
    s = train[c]
    if (s.dtype == object or str(s.dtype) in {"string", "category"}) and s.nunique(dropna=True) == len(s):
        feature_cols.remove(c)

with open(paths["latest_model"], "rb") as f:
    model = pickle.load(f)

def final_estimator(obj):
    return obj.steps[-1][1] if hasattr(obj, "steps") else obj

def fmt_value(v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4g}"
    return str(v)

est = final_estimator(model)
model_class = model.__class__.__name__ if not hasattr(model, "steps") else f"{model.__class__.__name__}(" + " -> ".join(n for n, _ in model.steps) + ")"
params = est.get_params(deep=False) if hasattr(est, "get_params") else {}
hyper_parts = []
for k, v in params.items():
    if isinstance(v, (str, int, float, bool, type(None))):
        hyper_parts.append(f"{k}={fmt_value(v)}")
    else:
        hyper_parts.append(f"{k}={v.__class__.__name__}")
if hasattr(est, "estimator") and hasattr(est.estimator, "get_params"):
    inner = est.estimator.get_params(deep=False)
    hyper_parts.append("inner_estimator=" + est.estimator.__class__.__name__)
    hyper_parts.extend(f"inner_{k}={fmt_value(v) if isinstance(v, (str, int, float, bool, type(None))) else v.__class__.__name__}" for k, v in inner.items())
hyperparams = "; ".join(hyper_parts[:18])

def importance_vector(final):
    imp = getattr(final, "feature_importances_", None)
    if imp is not None:
        return np.asarray(imp)
    if hasattr(final, "coef_"):
        coef = np.asarray(final.coef_)
        return np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    if hasattr(final, "estimators_"):
        vals = []
        for sub in final.estimators_:
            sub_imp = getattr(sub, "feature_importances_", None)
            if sub_imp is not None:
                vals.append(np.asarray(sub_imp))
            elif hasattr(sub, "coef_"):
                coef = np.asarray(sub.coef_)
                vals.append(np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef))
        if vals:
            return np.mean(vals, axis=0)
    return None

imp = importance_vector(est)
top_feature = "unavailable"
if imp is not None:
    names = None
    if hasattr(model, "feature_names_in_") and len(model.feature_names_in_) == len(imp):
        names = list(model.feature_names_in_)
    elif len(feature_cols) == len(imp):
        names = feature_cols
    if names is None:
        try:
            with open(paths["06_features_retain_features_3E3KFT0A/selector_all.pkl"], "rb") as f:
                sel = pickle.load(f)
            if hasattr(sel, "get_feature_names_out"):
                names = list(sel.get_feature_names_out())
        except Exception:
            names = None
    if names is not None and len(names) == len(imp):
        top_feature = str(names[int(np.argmax(imp))])

vm = pd.read_csv(paths["09_validate_score_validation_F12RYJBQ/validation_metrics.csv"])
rows = [
    {"aspect": "task", "detail": f"multioutput_regression; targets={targets}; primary_metric={primary_metric}", "source": "_protocol"},
    {"aspect": "frozen_model", "detail": f"{model_class}; final_estimator={est.__class__.__name__}", "source": "latest_model"},
    {"aspect": "selected_hyperparameters", "detail": hyperparams if hyperparams else "none available from saved estimator", "source": "latest_model"},
    {"aspect": "input_feature_count", "detail": str(len(feature_cols)), "source": "train"},
    {"aspect": "input_features", "detail": ", ".join(map(str, feature_cols)), "source": "train"},
]
for i, row in vm.head(12).iterrows():
    detail = "; ".join(f"{c}={fmt_value(row[c])}" for c in vm.columns if pd.notna(row[c]))
    rows.append({"aspect": f"validation_evidence_{i}", "detail": detail, "source": "09_validate_score_validation_F12RYJBQ/validation_metrics.csv"})
rows.append({"aspect": "limitations", "detail": "Frozen saved model interpreted without refitting; validation evidence comes from the provided metrics file; feature ranking is model-specific; no causal interpretation; unavailable importance is reported explicitly.", "source": "interpretation.csv"})

out = pd.DataFrame(rows)
mlca_runtime.save_table(out, "interpretation.csv", caption="Frozen model interpretation and validation evidence")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print("features", len(feature_cols))
print("interpretation_rows", len(out))
print("top_feature", top_feature)
print("validation_rows", len(vm))
