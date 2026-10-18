import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())

try:
    targets = list(proto["targets"])
    excluded = list(proto["excluded_columns"])
    group = proto["group_column"]
    metric = proto["task"]["primary_metric"]
except KeyError as e:
    raise KeyError(f"Missing required protocol key {e}; available top-level keys: {sorted(proto.keys())}")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def keylike(col):
    s = train[col]
    if s.is_unique and (pd.api.types.is_object_dtype(s) or pd.api.types.is_integer_dtype(s)):
        return True
    return False

feature_cols = [c for c in train.columns if c not in set(targets) | set(excluded) | {group} and not keylike(c)]

def load_pickle(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

def load_json(key):
    try:
        return json.loads(inp[key].read_text())
    except Exception:
        return {}

pre = load_pickle("06_preprocess_fit_preprocessing_DB5T0R19/preprocessor.pkl")
sel = load_pickle("07_features_retain_features_05E7EJVW/selector_all.pkl")
model = load_pickle("08_model_train_baseline_RDZRN5AQ/model.pkl")
alt = load_pickle("09_model_train_alternative_J8YMJGXB/alternative_model.pkl")
base_man = load_json("08_model_train_baseline_RDZRN5AQ/manifest.json")
alt_man = load_json("09_model_train_alternative_J8YMJGXB/manifest.json")

def final_est(obj):
    if hasattr(obj, "named_steps"):
        return list(obj.named_steps.values())[-1]
    return obj

def out_names(obj):
    if obj is None:
        return None
    try:
        if hasattr(obj, "get_feature_names_out"):
            return [str(x) for x in obj.get_feature_names_out()]
    except Exception:
        pass
    if hasattr(obj, "named_steps"):
        for step in reversed(list(obj.named_steps.values())):
            try:
                if hasattr(step, "get_feature_names_out"):
                    return [str(x) for x in step.get_feature_names_out()]
            except Exception:
                pass
            if hasattr(step, "feature_names_in_"):
                return [str(x) for x in step.feature_names_in_]
    if hasattr(obj, "feature_names_in_"):
        return [str(x) for x in obj.feature_names_in_]
    return None

def importances(est):
    if hasattr(est, "feature_importances_"):
        return np.asarray(est.feature_importances_).ravel()
    if hasattr(est, "coef_"):
        c = np.asarray(est.coef_)
        return np.abs(c) if c.ndim == 1 else np.abs(c).mean(axis=0)
    if hasattr(est, "estimators_"):
        arr = []
        for sub in est.estimators_:
            a = importances(sub)
            if a is not None:
                arr.append(a)
        if arr:
            return np.mean(arr, axis=0)
    return None

def params_text(man, obj):
    for k in ("selected_hyperparameters", "best_params", "hyperparameters", "params"):
        if isinstance(man, dict) and k in man:
            return json.dumps(man[k], default=str)[:600]
    est = final_est(obj)
    try:
        p = est.get_params(deep=False)
    except Exception:
        p = {}
    p = {k: v for k, v in p.items() if isinstance(v, (str, int, float, bool, type(None)))}
    return json.dumps(p, default=str)[:600]

est = final_est(model)
vals = importances(est)
names = out_names(est) or out_names(sel) or out_names(pre)
if vals is not None and (names is None or len(names) != len(vals)):
    names = feature_cols[: len(vals)]
top_feature = names[int(np.argmax(vals))] if vals is not None and len(vals) and names else "unknown"

vm = pd.read_csv(inp["10_validate_score_validation_3J1WH09P/validation_metrics.csv"])
metric_val = None
cols = {c.lower(): c for c in vm.columns}
if "metric" in cols and "value" in cols:
    m = vm[cols["metric"]].astype(str).str.lower() == str(metric).lower()
    if m.any():
        metric_val = float(vm.loc[m, cols["value"]].iloc[0])
if metric_val is None:
    for c in vm.columns:
        if str(metric).lower() in c.lower():
            s = pd.to_numeric(vm[c], errors="coerce").dropna()
            if len(s):
                metric_val = float(s.iloc[0])
                break
if metric_val is None:
    for c in vm.columns:
        s = pd.to_numeric(vm[c], errors="coerce").dropna()
        if len(s):
            metric_val = float(s.iloc[0])
            break

rows = [
    ("primary_metric", metric, "_protocol"),
    ("targets", ", ".join(targets), "_protocol"),
    ("group_column", group, "_protocol"),
    ("n_train_rows", len(train), "train"),
    ("n_val_rows", len(val), "val"),
    ("n_features", len(feature_cols), "train"),
    ("model_type", type(model).__name__, "08_model_train_baseline_RDZRN5AQ/model.pkl"),
    ("final_estimator", type(est).__name__, "08_model_train_baseline_RDZRN5AQ/model.pkl"),
    ("baseline_hyperparameters", params_text(base_man, model), "08_model_train_baseline_RDZRN5AQ/manifest.json"),
    ("alternative_hyperparameters", params_text(alt_man, alt), "09_model_train_alternative_J8YMJGXB/manifest.json"),
    (f"validation_{metric}", "" if metric_val is None else round(metric_val, 4), "10_validate_score_validation_3J1WH09P/validation_metrics.csv"),
    ("top_feature", top_feature, "08_model_train_baseline_RDZRN5AQ/model.pkl"),
    ("limitations", "Frozen interpretation only; no refitting or test exposure. Feature importance is model-derived, not causal. Nested taxonomic labels and RecordID grouping limit independence assumptions.", "_protocol"),
]
interp = pd.DataFrame(rows, columns=["item", "value", "source"])
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation summary")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"rows={len(interp)} features={len(feature_cols)} top_feature={top_feature} val_{metric}={metric_val}")
