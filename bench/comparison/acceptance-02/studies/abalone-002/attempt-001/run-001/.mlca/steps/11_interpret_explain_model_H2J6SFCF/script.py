import mlca_runtime, json, pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
I = mlca_runtime.inputs()

proto = json.loads(I["_protocol"].read_text())
need = ["targets", "excluded_columns", "group_column", "task"]
miss = [k for k in need if k not in proto]
if miss:
    raise KeyError(f"Missing protocol keys {miss}; available top-level keys: {sorted(proto)}")
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group = proto["group_column"]
task = proto["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; task keys: {sorted(task)}")
primary_metric = task["primary_metric"]

train = pd.read_parquet(I["train"])
val = pd.read_parquet(I["val"])

def is_key_like(s):
    n, nu = len(s), s.nunique(dropna=True)
    if n == 0 or nu != n:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        nm = str(s.name).lower()
        return any(t in nm for t in ("id", "key", "index", "row", "code"))
    return False

group_cols = [] if group is None or (isinstance(group, float) and pd.isna(group)) else [group]
if group_cols and group not in train.columns:
    raise KeyError(f"group_column {group!r} not in train columns")
drop = set(targets + excluded + group_cols)
feature_cols = [c for c in train.columns if c not in drop and not is_key_like(train[c])]

def load_pkl(key):
    with open(I[key], "rb") as f:
        return pickle.load(f)

def load_json(key):
    return json.loads(I[key].read_text())

def final_est(est):
    if hasattr(est, "named_steps") and len(est.named_steps):
        return list(est.named_steps.values())[-1]
    if hasattr(est, "steps") and len(est.steps):
        return est.steps[-1][1]
    return est

def desc(est):
    d = type(est).__name__
    if hasattr(est, "named_steps"):
        d += f" (final: {type(final_est(est)).__name__})"
    return d

def params_text(est, manifest):
    for k in ("params", "hyperparameters", "best_params", "model_params",
              "selected_params", "estimator_params", "best_hyperparameters", "tuned_params"):
        if isinstance(manifest.get(k), dict):
            return json.dumps(manifest[k], default=str, sort_keys=True)[:1000]
    try:
        return json.dumps(final_est(est).get_params(deep=False), default=str, sort_keys=True)[:1000]
    except Exception:
        return "unavailable"

def imp_top(est):
    fe = final_est(est)
    imp = getattr(fe, "feature_importances_", None)
    if imp is None:
        coef = getattr(fe, "coef_", None)
        if coef is not None:
            imp = np.abs(np.ravel(coef))
    if imp is None:
        return None
    names = None
    if hasattr(est, "named_steps"):
        try:
            names = list(final_est(est).feature_names_in_)
        except Exception:
            names = None
        if names is None:
            try:
                names = list(est[:-1].get_feature_names_out())
            except Exception:
                names = None
    if names is None:
        names = getattr(fe, "feature_names_in_", None)
        names = list(names) if names is not None else None
    imp = np.ravel(imp)
    if names is None or len(names) != len(imp):
        return None
    return str(names[int(np.argmax(imp))])

def manifest_top(m):
    for k in ("top_feature", "top_features", "selected_features", "features",
              "feature_names", "retained_features"):
        v = m.get(k)
        if isinstance(v, list) and v:
            return str(v[0])
        if isinstance(v, str):
            return v
    return None

base = load_pkl("07_model_train_baseline_1RE4HZWD/model.pkl")
latest = load_pkl("latest_model")
pre = load_pkl("05_preprocess_fit_preprocessing_NRNQX0TZ/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_YTKXSJV5/selector_all.pkl")
base_m = load_json("07_model_train_baseline_1RE4HZWD/manifest.json")
alt_m = load_json("08_model_train_alternative_22ZP3DFW/manifest.json")

top_feature = imp_top(latest) or imp_top(base) or manifest_top(alt_m) or manifest_top(base_m) or "unavailable"

def n_out(obj):
    try:
        return len(obj.get_feature_names_out())
    except Exception:
        return getattr(obj, "n_features_in_", None)

rows = []
def add(sec, item, text=None, num=None, src=""):
    rows.append((sec, item, text if text is not None else np.nan,
                 float(f"{float(num):.4g}") if num is not None and pd.notna(num) else np.nan, src))

add("protocol", "targets", ", ".join(map(str, targets)), None, "_protocol")
add("protocol", "excluded_columns", ", ".join(map(str, excluded)) if excluded else "none", None, "_protocol")
add("protocol", "group_column", "none" if not group_cols else str(group), None, "_protocol")
add("protocol", "primary_metric", primary_metric, None, "_protocol")
add("input", "n_train_rows", None, len(train), "train")
add("input", "n_val_rows", None, len(val), "val")
add("input", "n_feature_columns", None, len(feature_cols), "train")
add("input", "feature_columns", ", ".join(feature_cols), None, "train")
add("input", "preprocessor", desc(pre), None, "05_preprocess_fit_preprocessing_NRNQX0TZ/preprocessor.pkl")
add("input", "preprocessor_output_features", None, n_out(pre), "05_preprocess_fit_preprocessing_NRNQX0TZ/preprocessor.pkl")
add("input", "selector", desc(sel), None, "06_features_retain_features_YTKXSJV5/selector_all.pkl")
add("input", "selector_output_features", None, n_out(sel), "06_features_retain_features_YTKXSJV5/selector_all.pkl")
add("model", "baseline_model", desc(base), None, "07_model_train_baseline_1RE4HZWD/model.pkl")
add("model", "baseline_hyperparameters", params_text(base, base_m), None, "07_model_train_baseline_1RE4HZWD/manifest.json")
add("model", "latest_model", desc(latest), None, "latest_model")
add("model", "latest_hyperparameters", params_text(latest, alt_m), None, "08_model_train_alternative_22ZP3DFW/manifest.json")
add("model", "top_feature", top_feature, None, "interpretation.csv")

def metric_rows(path, src):
    try:
        m = pd.read_csv(path)
    except Exception:
        return []
    out = []
    def nv(v):
        try:
            f = float(v)
            return f if pd.notna(f) else None
        except Exception:
            return None
    if m.empty:
        return out
    if m.shape[0] == 1:
        for c in m.columns:
            v = nv(m.iloc[0][c])
            if v is not None:
                out.append(("validation", str(c), None, v, src))
    else:
        mcol = next((c for c in m.columns if str(c).lower() in ("metric", "name", "metric_name")), None)
        vcol = next((c for c in m.columns if str(c).lower() in ("value", "score", "metric_value")), None)
        if mcol and vcol:
            for _, r in m.head(20).iterrows():
                v = nv(r[vcol])
                if v is not None:
                    item = str(r[mcol])
                    if "model" in m.columns:
                        item = f"{r['model']}:{item}"
                    out.append(("validation", item, None, v, src))
        else:
            numcols = list(m.select_dtypes(include="number").columns)
            for i, r in m.head(10).iterrows():
                for c in numcols[:4]:
                    v = nv(r[c])
                    if v is not None:
                        out.append(("validation", f"{c}[{i}]", None, v, src))
    return out

for r in metric_rows(I["09_validate_score_validation_RP1BFKWM/validation_metrics.csv"],
                     "09_validate_score_validation_RP1BFKWM/validation_metrics.csv"):
    add(*r)

add("limitations", "no_refit", "No model fitting, selection, or threshold tuning performed in interpretation.", None, "interpretation.csv")
add("limitations", "validation_only", "Validation metrics are held-out evidence; no test data used.", None, "09_validate_score_validation_RP1BFKWM/validation_metrics.csv")
add("limitations", "single_target", "Only Rings is modeled; count outcome may favor Poisson/log1p alternatives not evaluated here.", None, "_protocol")
add("limitations", "no_groups", "No repeated-subject or group structure declared; rows treated as independent.", None, "_protocol")
add("limitations", "feature_scope", "Feature list excludes targets, excluded columns, group column, and key-like columns from frozen train table.", None, "train")

rows = rows[:50]
out = pd.DataFrame(rows, columns=["section", "item", "value_text", "numeric_value", "source_file"])
mlca_runtime.save_table(out, "interpretation.csv", caption="Frozen model interpretation summary")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"train_rows={len(train)} val_rows={len(val)} features={len(feature_cols)}")
print(f"top_feature={top_feature}")
print(f"interpretation_rows={len(out)}")
