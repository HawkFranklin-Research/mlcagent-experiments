import mlca_runtime, pickle, json, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing {k!r}. Available top-level keys: {sorted(d.keys())}")
    return d[k]

protocol = json.loads(inputs["_protocol"].read_text())
targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
group = req(protocol, "group_column")
task = req(protocol, "task")
primary = req(task, "primary_metric")

train = pd.read_parquet(inputs["train"])

def key_like(s, name):
    if len(s) == 0 or s.nunique(dropna=True) != len(s):
        return False
    if pd.api.types.is_float_dtype(s.dtype):
        return False
    low = name.lower()
    if any(t in low for t in ("id", "key", "code")):
        return True
    return not pd.api.types.is_numeric_dtype(s.dtype)

feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group and not key_like(train[c], c)
]

def load_pkl(p):
    with p.open("rb") as f:
        return pickle.load(f)

model = load_pkl(inputs["latest_model"])
baseline = load_pkl(inputs["07_model_train_baseline_JP4JWWDQ/model.pkl"])
pre = load_pkl(inputs["05_preprocess_fit_preprocessing_A8X1QVXT/preprocessor.pkl"])
sel = load_pkl(inputs["06_features_retain_features_ZGB5PBQC/selector_all.pkl"])

def final_est(o):
    return o.steps[-1][1] if isinstance(o, Pipeline) and o.steps else o

def base_est(e):
    return e.estimator if e.__class__.__name__ == "MultiOutputClassifier" and hasattr(e, "estimator") else e

def step_summary(o):
    if isinstance(o, Pipeline):
        return " -> ".join(f"{k}:{v.__class__.__name__}" for k, v in o.named_steps.items())
    return o.__class__.__name__

def imp_vec(o):
    if o is None:
        return None
    for attr in ("feature_importances_", "coef_", "scores_"):
        if hasattr(o, attr):
            try:
                v = np.asarray(getattr(o, attr), dtype=float)
                if attr == "coef_" and v.ndim > 1:
                    v = np.mean(np.abs(v), axis=0)
                else:
                    v = np.abs(v).ravel()
                if v.size:
                    return v
            except Exception:
                pass
    if hasattr(o, "estimators_"):
        vs = [imp_vec(e) for e in o.estimators_]
        vs = [v for v in vs if v is not None]
        if vs:
            try:
                return np.mean(vs, axis=0)
            except Exception:
                pass
    if hasattr(o, "estimator"):
        return imp_vec(o.estimator)
    return None

def names_for(o, v, feat_cols, prep, selector):
    n = len(v)
    cands = []
    if isinstance(o, Pipeline):
        try:
            cands.append(list(o[:-1].get_feature_names_out(feat_cols)))
        except Exception:
            try:
                cands.append(list(o[:-1].get_feature_names_out()))
            except Exception:
                pass
    e = final_est(o)
    for obj in (e, o, selector, prep):
        if obj is not None and hasattr(obj, "feature_names_in_"):
            try:
                cands.append(list(obj.feature_names_in_))
            except Exception:
                pass
    for c in cands:
        if len(c) == n:
            return c
    if len(feat_cols) == n:
        return feat_cols
    for obj in (selector, prep):
        if obj is not None and hasattr(obj, "get_support") and hasattr(obj, "feature_names_in_"):
            try:
                s = obj.get_support()
                if int(np.sum(s)) == n:
                    return list(np.asarray(obj.feature_names_in_)[s])
            except Exception:
                pass
    return None

top = None
for obj in (model, sel, pre):
    vec = imp_vec(obj)
    if vec is None:
        continue
    names = names_for(obj, vec, feature_cols, pre, sel)
    if names is not None and len(names) == len(vec) and len(vec):
        top = str(names[int(np.argmax(vec))])
        break

def fmt(v):
    if v is None:
        return "NaN"
    if isinstance(v, (np.floating, float)):
        return f"{float(v):.4g}"
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return str(int(v))
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (list, tuple)):
        if len(v) <= 50:
            return ", ".join(fmt(x) for x in v)
        return f"{len(v)} items"
    if isinstance(v, dict):
        return f"{len(v)} keys"
    s = str(v).replace("\n", " ")
    return s[:400] + ("..." if len(s) > 400 else "")

src_model = "latest_model"
rows = []
def add(sec, item, val, src):
    rows.append({"section": sec, "item": item, "value": fmt(val), "source_file": src})

add("task", "task_type", task.get("type", ""), "_protocol")
add("task", "primary_metric", primary, "_protocol")
add("task", "targets", targets, "_protocol")
add("task", "group_column", group, "_protocol")
add("task", "excluded_columns", excluded, "_protocol")
add("inputs", "feature_count", len(feature_cols), "train")
add("inputs", "feature_names", feature_cols, "train")
add("model", "frozen_source", "latest_model", src_model)
add("model", "frozen_class", model.__class__.__name__, src_model)
add("model", "frozen_steps", step_summary(model), src_model)
add("model", "frozen_final_estimator", final_est(model).__class__.__name__, src_model)
be = base_est(final_est(model))
if be is not final_est(model):
    add("model", "multioutput_base_estimator", be.__class__.__name__, src_model)
add("model", "baseline_class", baseline.__class__.__name__, "07_model_train_baseline_JP4JWWDQ/model.pkl")
add("preprocessing", "preprocessor_steps", step_summary(pre), "05_preprocess_fit_preprocessing_A8X1QVXT/preprocessor.pkl")
add("preprocessing", "selector_class", sel.__class__.__name__, "06_features_retain_features_ZGB5PBQC/selector_all.pkl")
if hasattr(sel, "get_support"):
    try:
        add("preprocessing", "selector_support_count", int(np.sum(sel.get_support())), "06_features_retain_features_ZGB5PBQC/selector_all.pkl")
    except Exception:
        pass
if hasattr(model, "feature_names_in_"):
    add("model", "model_feature_names_in", list(model.feature_names_in_), src_model)
if top:
    add("interpretation", "top_feature", top, src_model)

params = []
def addp(item, val):
    if len(params) < 14:
        params.append({"section": "hyperparameters", "item": item, "value": fmt(val), "source_file": src_model})

est = final_est(model)
be = base_est(est)
addp("final_class", est.__class__.__name__)
if be is not est:
    addp("base_class", be.__class__.__name__)
for prefix, e in (("final", est), ("base", be)):
    if prefix == "base" and e is est:
        continue
    try:
        ps = e.get_params(deep=False) if hasattr(e, "get_params") else {}
    except Exception:
        ps = {}
    for k, v in ps.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            addp(f"{prefix}.{k}", v)
        elif isinstance(v, (tuple, list)) and len(v) <= 5 and all(isinstance(x, (str, int, float, bool, type(None))) for x in v):
            addp(f"{prefix}.{k}", str(v))

val_rows = []
try:
    vd = pd.read_csv(inputs["09_validate_score_validation_6A3SAZRV/validation_metrics.csv"])
    def addv(item, val):
        val_rows.append({"section": "validation", "item": item, "value": fmt(val), "source_file": "09_validate_score_validation_6A3SAZRV/validation_metrics.csv"})
    addv("validation_rows", len(vd))
    mcol = next((c for c in vd.columns if c.lower() in ("metric", "name", "key")), None)
    vcol = next((c for c in vd.columns if c.lower() in ("value", "score", "mean", "metric_value")), None)
    if mcol is not None and vcol is not None:
        tmp = vd.copy()
        tmp["_p"] = tmp[mcol].astype(str).eq(str(primary))
        tmp = tmp.sort_values("_p", ascending=False)
        for _, r in tmp.head(10).iterrows():
            addv(str(r[mcol]), r[vcol])
    else:
        for c in vd.columns[:10]:
            if pd.api.types.is_numeric_dtype(vd[c]):
                addv(c, vd[c].iloc[0] if len(vd) == 1 else vd[c].mean())
except Exception as e:
    val_rows.append({"section": "validation", "item": "validation_metrics_error", "value": type(e).__name__, "source_file": "09_validate_score_validation_6A3SAZRV/validation_metrics.csv"})

lim_rows = [
    {"section": "limitations", "item": "grouping", "value": "RecordID groups call segments from the same recording; validation rows are not fully independent.", "source_file": "interpretation.csv"},
    {"section": "limitations", "item": "targets", "value": "Family, Genus and Species are nested taxonomic labels; coarse and fine label errors are correlated.", "source_file": "interpretation.csv"},
    {"section": "limitations", "item": "evidence", "value": "Validation evidence is limited to the held-out validation split; no test or refitting is performed here.", "source_file": "09_validate_score_validation_6A3SAZRV/validation_metrics.csv"},
    {"section": "limitations", "item": "scope", "value": "Interpretation is descriptive; feature importance does not imply causality.", "source_file": "interpretation.csv"},
]

final_rows = rows + params + val_rows + lim_rows
if len(final_rows) > 50:
    final_rows = final_rows[:49]
    final_rows.append({"section": "meta", "item": "truncated", "value": "more than 50 rows", "source_file": "interpretation.csv"})

out = pd.DataFrame(final_rows, columns=["section", "item", "value", "source_file"])
mlca_runtime.save_table(out, "interpretation.csv", caption="Frozen multilabel model interpretation summary")
if top:
    mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")

print(f"interpretation rows: {len(out)}")
print(f"features: {len(feature_cols)}")
if top:
    print(f"top_feature: {top}")
