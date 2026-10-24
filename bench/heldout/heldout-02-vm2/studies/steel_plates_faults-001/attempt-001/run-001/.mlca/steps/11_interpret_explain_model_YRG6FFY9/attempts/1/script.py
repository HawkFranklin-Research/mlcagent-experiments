import mlca_runtime, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

def read_json(p):
    return json.loads(Path(p).read_text())

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

proto = read_json(inp["_protocol"])

def pget(path):
    cur = proto
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing {k!r} at {path}; available top-level keys: {list(proto.keys())}")
        cur = cur[k]
    return cur

targets = list(pget(["targets"]))
excluded = list(pget(["excluded_columns"]))
group = pget(["group_column"])
primary_metric = pget(["task", "primary_metric"])
task = pget(["task"])

model_keys = ["latest_model", "07_model_train_baseline_HK97MR3S/model.pkl",
              "08_model_train_alternative_2D5YHWB5/alternative_model.pkl"]
model_key = next(k for k in model_keys if k in inp and Path(inp[k]).exists())
model = load_pkl(inp[model_key])

def maybe_pkl(suffix):
    k = next((x for x in inp if x.endswith(suffix)), None)
    return load_pkl(inp[k]) if k else None

pre = maybe_pkl("preprocessor.pkl")
sel = maybe_pkl("selector_all.pkl")

def names_out(obj):
    if obj is None:
        return None
    try:
        return [str(x) for x in np.asarray(obj.get_feature_names_out()).ravel()]
    except Exception:
        pass
    if hasattr(obj, "feature_names_in_"):
        try:
            return [str(x) for x in np.asarray(obj.feature_names_in_).ravel()]
        except Exception:
            pass
    return None

def final_est(est):
    if hasattr(est, "named_steps"):
        return list(est.named_steps.values())[-1]
    return est

def importance_vector(m):
    roots = list(m.estimators_) if hasattr(m, "estimators_") else [m]
    if not hasattr(m, "estimators_") and hasattr(m, "named_steps"):
        fin = final_est(m)
        roots = list(fin.estimators_) if hasattr(fin, "estimators_") else [m]
    vecs = []
    for r in roots:
        e = final_est(r)
        if hasattr(e, "coef_"):
            c = np.abs(np.asarray(e.coef_, dtype=float))
            if c.ndim > 1:
                c = c.mean(axis=0)
            vecs.append(np.ravel(c))
        elif hasattr(e, "feature_importances_"):
            vecs.append(np.ravel(np.asarray(e.feature_importances_, dtype=float)))
    if not vecs or any(len(v) != len(vecs[0]) for v in vecs):
        return None
    return np.mean(np.vstack(vecs), axis=0)

def importance_names(m, n):
    if hasattr(m, "named_steps"):
        steps = list(m.named_steps.items())
        fin = steps[-1][1]
        if hasattr(fin, "feature_names_in_"):
            nm = [str(x) for x in np.asarray(fin.feature_names_in_).ravel()]
            if len(nm) == n:
                return nm
        for _, step in reversed(steps[:-1]):
            nm = names_out(step)
            if nm is not None and len(nm) == n:
                return nm
    nm = names_out(m)
    if nm is not None and len(nm) == n:
        return nm
    if hasattr(m, "feature_names_in_"):
        nm = [str(x) for x in np.asarray(m.feature_names_in_).ravel()]
        if len(nm) == n:
            return nm
    pre_nm = names_out(pre)
    if sel is not None:
        sn = names_out(sel)
        if sn is not None and len(sn) == n:
            return sn
        if pre_nm is not None and hasattr(sel, "get_support"):
            try:
                sup = sel.get_support()
                if len(sup) == len(pre_nm):
                    sn = [x for x, s in zip(pre_nm, sup) if s]
                    if len(sn) == n:
                        return sn
            except Exception:
                pass
    return pre_nm if pre_nm is not None and len(pre_nm) == n else None

def is_keylike(s, name):
    n, nu = len(s), s.nunique(dropna=True)
    if n <= 50:
        return False
    low = str(name).lower()
    if s.dtype == object or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        return nu == n
    if pd.api.types.is_integer_dtype(s.dtype):
        return nu == n and any(t in low for t in ["id", "index", "code", "key"])
    return False

def feature_cols(df):
    g = {group} if isinstance(group, str) else set(group or [])
    drop = set(targets) | set(excluded) | g
    return [c for c in df.columns if c not in drop and not is_keylike(df[c], c)]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
raw_feats = feature_cols(train)
model_nm = names_out(model)
if model_nm is None and hasattr(model, "feature_names_in_"):
    model_nm = [str(x) for x in np.asarray(model.feature_names_in_).ravel()]
if model_nm is None:
    model_nm = raw_feats

imp = importance_vector(model)
top_name, top_val, imp_nm = None, np.nan, None
if imp is not None:
    imp_nm = importance_names(model, len(imp))
    if imp_nm is not None:
        i = int(np.nanargmax(imp))
        top_name, top_val = imp_nm[i], float(imp[i])

vm = pd.read_csv(inp["09_validate_score_validation_C1P5YKW9/validation_metrics.csv"])
recs = []
cols = {c.lower(): c for c in vm.columns}
if "metric" in cols and "value" in cols:
    for _, r in vm.iterrows():
        try:
            recs.append((str(r[cols["metric"]]), float(r[cols["value"]])))
        except Exception:
            pass
elif "name" in cols and "value" in cols:
    for _, r in vm.iterrows():
        try:
            recs.append((str(r[cols["name"]]), float(r[cols["value"]])))
        except Exception:
            pass
elif len(vm) == 1:
    for c in vm.columns:
        try:
            recs.append((str(c), float(vm.iloc[0][c])))
        except Exception:
            pass
prim_val = next((v for n, v in recs if primary_metric.lower() in n.lower()), np.nan)
if not np.isfinite(prim_val) and len(recs) == 1:
    prim_val = recs[0][1]
if np.isfinite(prim_val):
    mlca_runtime.record_metric(f"val_{primary_metric}", float(prim_val), "val")

def fmt(v):
    if v is None:
        return np.nan
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return np.nan if not np.isfinite(v) else float(f"{v:.4g}")
    if isinstance(v, (list, tuple)):
        return "; ".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, default=str)
    return v

rows = []
def add(section, item, value, source=""):
    rows.append({"section": section, "item": item, "value": fmt(value), "source": source})

fin = final_est(model)
add("model", "frozen_model_file", Path(inp[model_key]).name, Path(inp[model_key]).name)
add("model", "model_class", type(model).__name__, Path(inp[model_key]).name)
if hasattr(model, "named_steps"):
    add("model", "pipeline_steps", list(model.named_steps.keys()), Path(inp[model_key]).name)
add("model", "final_estimator", type(fin).__name__, Path(inp[model_key]).name)
params = fin.get_params(deep=False) if hasattr(fin, "get_params") else {}
for k in ["C", "solver", "max_iter", "class_weight", "penalty", "n_estimators", "max_depth", "learning_rate"]:
    if k in params:
        add("hyperparameters", k, params[k], Path(inp[model_key]).name)
inner = getattr(fin, "estimator", None)
if inner is not None and hasattr(inner, "get_params"):
    ip = inner.get_params(deep=False)
    for k in ["C", "solver", "max_iter", "class_weight", "penalty"]:
        if k in ip:
            add("hyperparameters", f"inner_{k}", ip[k], Path(inp[model_key]).name)
add("inputs", "targets", targets, Path(inp["_protocol"]).name)
add("inputs", "raw_feature_count", len(raw_feats), "train.parquet")
add("inputs", "model_input_feature_count", len(model_nm), Path(inp[model_key]).name)
add("inputs", "raw_features_preview", raw_feats[:20], "train.parquet")
add("validation", "primary_metric", primary_metric, Path(inp["_protocol"]).name)
add("validation", f"val_{primary_metric}", prim_val, Path(inp["09_validate_score_validation_C1P5YKW9/validation_metrics.csv"]).name)
add("validation", "validation_metrics_rows", len(vm), Path(inp["09_validate_score_validation_C1P5YKW9/validation_metrics.csv"]).name)
add("interpretation", "top_feature", top_name if top_name else np.nan, Path(inp[model_key]).name)
add("interpretation", "top_feature_importance", top_val, Path(inp[model_key]).name)
add("limitations", "no_refit", "Frozen model interpreted without fitting.", "")
add("limitations", "importance_method", "Model coefficients/feature_importances_ only; no permutation or refit.", Path(inp[model_key]).name)
add("limitations", "validation_scope", "Validation evidence is read-only; test data not used.", "")
add("limitations", "multilabel_targets", "Seven binary fault targets treated as joint multilabel outcomes.", Path(inp["_protocol"]).name)

interp = pd.DataFrame(rows)
mlca_runtime.save_table(interp, "interpretation.csv")
if top_name is not None and np.isfinite(top_val):
    mlca_runtime.record_fact_candidate("top_feature", str(top_name), source="interpretation.csv")

print(f"train={train.shape} val={val.shape}")
print(f"model={type(model).__name__} file={Path(inp[model_key]).name}")
print(f"primary_metric={primary_metric} val={prim_val}")
print(f"top_feature={top_name}")
print(f"saved interpretation.csv rows={len(interp)}")
