import mlca_runtime, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
outd = mlca_runtime.output_dir()

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def load_json(p):
    return json.loads(Path(p).read_text())

protocol = load_json(inp["_protocol"])
task = protocol["task"]
primary = task["primary_metric"]
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column", None)

train = pd.read_parquet(inp["train"])
drop = set(map(str, targets)) | set(map(str, excluded))
if group is not None:
    drop.add(str(group))

feature_cols = []
n_train = len(train)
for c in train.columns:
    if str(c) in drop:
        continue
    s = train[c]
    lname = str(c).lower()
    if (s.dtype == object or str(s.dtype).startswith("string") or str(s.dtype) == "category") and s.nunique(dropna=True) == n_train:
        continue
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=True) == n_train and ("id" in lname or "key" in lname or "seq" in lname):
        continue
    feature_cols.append(c)

pre = load_pkl(inp["05_preprocess_fit_preprocessing_4ZZYYY8Y/preprocessor.pkl"])
sel = load_pkl(inp["06_features_retain_features_E2FAAE9F/selector_all.pkl"])
model = load_pkl(inp["latest_model"])
baseline = load_pkl(inp["07_model_train_baseline_Q9V0XT19/model.pkl"])

def final_est(m):
    while hasattr(m, "steps") and len(m.steps) > 0:
        m = m.steps[-1][1]
    return m

def class_desc(o):
    if hasattr(o, "steps"):
        return type(o).__name__ + ": " + ", ".join(n for n, _ in o.steps)
    if hasattr(o, "transformers"):
        return type(o).__name__ + ": " + ", ".join(str(t[0]) for t in o.transformers)
    return type(o).__name__

def fmt_num(x):
    if x is None:
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        return f"{float(x):.4g}"
    except Exception:
        return str(x)

def fmt_params(est):
    keys = ["n_components", "whiten", "svd_solver", "hidden_layer_sizes", "activation",
            "solver", "alpha", "learning_rate_init", "max_iter", "tol", "batch_size",
            "random_state", "n_neighbors", "contamination"]
    p = est.get_params(deep=False) if hasattr(est, "get_params") else {}
    chosen = [k for k in keys if k in p]
    if not chosen:
        chosen = [k for k in p if not str(k).startswith("_")][:8]
    return "; ".join(f"{k}={p[k]}" for k in chosen)[:300]

def input_names_for_final(m, base_names):
    names = list(base_names)
    if hasattr(m, "steps"):
        for _, step in m.steps[:-1]:
            if step is None or (isinstance(step, str) and step == "passthrough"):
                continue
            try:
                out = step.get_feature_names_out(names)
            except Exception:
                try:
                    out = step.get_feature_names_out()
                except Exception:
                    continue
            try:
                names = [str(x) for x in out]
            except Exception:
                pass
    est = final_est(m)
    n = getattr(est, "n_features_in_", None)
    if n is None or len(names) == n:
        return names
    for obj in [sel, pre]:
        try:
            out = obj.get_feature_names_out(base_names)
            out = [str(x) for x in out]
            if len(out) == n:
                return out
        except Exception:
            try:
                out = obj.get_feature_names_out()
                out = [str(x) for x in out]
                if len(out) == n:
                    return out
            except Exception:
                pass
    return names

def top_feature(m, base_names):
    est = final_est(m)
    names = input_names_for_final(m, base_names)
    idx = None
    if hasattr(est, "components_"):
        a = np.abs(np.asarray(est.components_))
        if a.ndim == 2 and a.shape[1] == len(names):
            idx = int(np.argmax(a[0]))
    if idx is None and hasattr(est, "feature_importances_"):
        a = np.abs(np.asarray(est.feature_importances_))
        if a.ndim == 1 and a.shape[0] == len(names):
            idx = int(np.argmax(a))
    if idx is None and hasattr(est, "coef_"):
        a = np.abs(np.asarray(est.coef_))
        if a.ndim == 1 and a.shape[0] == len(names):
            idx = int(np.argmax(a))
        elif a.ndim == 2:
            if a.shape[1] == len(names):
                idx = int(np.argmax(a[0]))
            elif a.shape[0] == len(names):
                idx = int(np.argmax(a[:, 0]))
    if idx is None and hasattr(est, "coefs_") and len(getattr(est, "coefs_", [])) > 0:
        W = np.asarray(est.coefs_[0])
        if W.ndim == 2 and W.shape[0] == len(names):
            idx = int(np.argmax(np.sum(np.abs(W), axis=1)))
    if idx is None or idx >= len(names):
        return "unknown"
    name = names[idx]
    if "__" in name and name.split("__")[-1] in base_names:
        name = name.split("__")[-1]
    return str(name)

tf = top_feature(model, feature_cols)
if tf == "unknown":
    tf = top_feature(baseline, feature_cols)

est = final_est(model)

try:
    vm = pd.read_csv(inp["09_validate_score_validation_TXYKFRBW/validation_metrics.csv"])
except Exception:
    vm = pd.read_parquet(inp["09_validate_score_validation_TXYKFRBW/validation_metrics.parquet"])

val = None
if primary in vm.columns:
    s = pd.to_numeric(vm[primary], errors="coerce").dropna()
    val = float(s.iloc[-1]) if len(s) else None
else:
    for c in vm.columns:
        if "metric" in str(c).lower():
            sub = vm[vm[c].astype(str).str.strip() == primary]
            if len(sub):
                vc = "value" if "value" in vm.columns else [x for x in vm.columns if x != c][0]
                s = pd.to_numeric(sub[vc], errors="coerce").dropna()
                if len(s):
                    val = float(s.iloc[0])
                    break
    if val is None:
        for c in vm.columns:
            if str(primary) in str(c) or str(c) in str(primary):
                s = pd.to_numeric(vm[c], errors="coerce").dropna()
                if len(s):
                    val = float(s.iloc[-1])
                    break

rows = []
def add(section, item, value, evidence=""):
    rows.append({"section": section, "item": item, "value": str(value), "evidence": str(evidence)})

add("model", "frozen model", class_desc(model), "latest_model")
add("model", "final estimator", type(est).__name__, "latest_model")
add("model", "hyperparameters", fmt_params(est), "latest_model")
add("model", "baseline comparison", type(final_est(baseline)).__name__, "07_model_train_baseline_Q9V0XT19/model.pkl")
add("inputs", "training rows", len(train), "train.parquet")
add("inputs", "feature columns", len(feature_cols), "train.parquet")
add("inputs", "preprocessor", class_desc(pre), "05_preprocess_fit_preprocessing_4ZZYYY8Y/preprocessor.pkl")
add("inputs", "selector", class_desc(sel), "06_features_retain_features_E2FAAE9F/selector_all.pkl")
add("inputs", "targets", str(list(targets)), "_protocol")
add("inputs", "excluded columns", str(list(excluded)), "_protocol")
add("validation", "primary metric", primary, "09_validate_score_validation_TXYKFRBW/validation_metrics.csv")
add("validation", "validation value", fmt_num(val), "09_validate_score_validation_TXYKFRBW/validation_metrics.csv")
add("validation", "validation rows", len(vm), "09_validate_score_validation_TXYKFRBW/validation_metrics.csv")
add("interpretation", "top feature (proxy)", tf, "latest_model")
add("limitations", "no outcome", "unsupervised representation; no label used", "_protocol")
add("limitations", "validation only", "no test exposure; metric is reconstruction error", "09_validate_score_validation_TXYKFRBW/validation_metrics.csv")
add("limitations", "top feature", "importance is a frozen-model proxy, not causal", "interpretation.csv")
add("limitations", "semantic meaning", "pixel loadings do not identify digit class", "interpretation.csv")

df = pd.DataFrame(rows).head(50)
mlca_runtime.save_table(df, "interpretation.csv", "Frozen model interpretation summary")

if val is not None:
    mlca_runtime.record_metric(primary, float(val), split="val")
mlca_runtime.record_fact_candidate("top_feature", tf, source="interpretation.csv")

print(f"train={train.shape}; features={len(feature_cols)}")
print(f"model={type(est).__name__}; top_feature={tf}")
if val is not None:
    print(f"{primary}={fmt_num(val)}")
print("saved interpretation.csv")
