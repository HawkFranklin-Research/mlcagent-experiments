import mlca_runtime, json, pickle, re
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from lifelines.statistics import proportional_hazard_test

mlca_runtime.set_seed(42)
P = mlca_runtime.inputs()
protocol = json.loads(Path(P["_protocol"]).read_text())
need = ["targets", "excluded_columns", "task"]
missing = [k for k in need if k not in protocol]
if missing:
    raise KeyError(f"missing {missing}; available top-level keys: {list(protocol.keys())}")
targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
task = protocol["task"]
primary = task["primary_metric"]

train = pd.read_parquet(P["train"])
val = pd.read_parquet(P["val"])

def key_like(s, c):
    if c in targets or c in excluded or (group and c == group):
        return True
    non = s.dropna()
    if len(non) == 0:
        return False
    if pd.api.types.is_integer_dtype(s):
        return non.nunique() == len(non)
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and non.nunique() == len(non)

feature_cols = [c for c in train.columns if not key_like(train[c], c)]

def load(k):
    with open(P[k], "rb") as f:
        return pickle.load(f)

def unwrap(o):
    seen = set()
    while id(o) not in seen:
        seen.add(id(o))
        if isinstance(o, dict):
            key = next((k for k in o if any(w in str(k).lower() for w in ("model", "estimator", "pipeline"))), None)
            if key is not None:
                o = o[key]
                continue
            vals = [v for v in o.values() if v is not None]
            if len(vals) == 1:
                o = vals[0]
                continue
            break
        if isinstance(o, (tuple, list)) and len(o) == 1:
            o = o[0]
            continue
        break
    return o

def expected_names(o):
    if o is None:
        return None
    if hasattr(o, "feature_names_in_"):
        return list(o.feature_names_in_)
    if hasattr(o, "params_"):
        return list(o.params_.index)
    return None

pre = load("05_preprocess_fit_preprocessing_M6JRF55T/preprocessor.pkl")
sel = load("06_features_select_features_64YJBEKC/selector.pkl")

def transform_with(s, p, X):
    sn = expected_names(s)
    if s is not None and sn is not None and all(c in X.columns for c in sn):
        return s.transform(X)
    Z = X
    if p is not None:
        Z = p.transform(Z)
    if s is not None:
        Z = s.transform(Z)
    return Z

def prepare_X(model, df):
    exp = expected_names(model)
    if exp is not None and all(c in df.columns for c in exp):
        return df[exp]
    if exp is None and hasattr(model, "named_steps"):
        return df[feature_cols]
    Z = transform_with(sel, pre, df[feature_cols])
    if exp is not None and hasattr(Z, "shape") and Z.shape[1] == len(exp):
        if isinstance(Z, pd.DataFrame):
            Z = Z.copy()
            Z.columns = exp
        else:
            Z = pd.DataFrame(Z, columns=exp, index=df.index)
    return Z

def final_est(model):
    steps = list(getattr(model, "named_steps", {}).items())
    return steps[-1][1] if steps else model

def to_frame(X, index):
    return X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, index=index)

def risk(model, X):
    model = unwrap(model)
    steps = list(getattr(model, "named_steps", {}).items())
    est, Xt = final_est(model), X
    if steps and len(steps) > 1:
        try:
            Xt = model[:-1].transform(X)
        except Exception:
            Xt = X
    names = expected_names(est)
    Xt = to_frame(Xt, getattr(X, "index", None))
    if names and Xt.shape[1] == len(names):
        Xt.columns = names
    if hasattr(est, "predict_partial_hazard"):
        return np.asarray(est.predict_partial_hazard(Xt)).ravel()
    if hasattr(est, "predict_log_partial_hazard"):
        return np.asarray(est.predict_log_partial_hazard(Xt)).ravel()
    if hasattr(est, "predict_expectation"):
        return -np.asarray(est.predict_expectation(Xt)).ravel()
    if hasattr(est, "predict_median"):
        return -np.asarray(est.predict_median(Xt)).ravel()
    if hasattr(est, "predict_proba"):
        return est.predict_proba(Xt)[:, 1]
    if hasattr(est, "predict"):
        return np.asarray(est.predict(Xt)).ravel()
    raise AttributeError(f"no risk prediction method on {type(est).__name__}")

def pick_outcomes():
    ev = next((c for c in targets if re.search(r"status|event|dead|death", c, re.I)), None)
    du = next((c for c in targets if c != ev and re.search(r"month|time|duration|survival", c, re.I)), None)
    if du is None:
        du = next((c for c in targets if c != ev), None)
    if ev is None:
        ev = next((c for c in targets if c != du), None)
    if du is None or ev is None:
        raise KeyError(f"cannot determine duration/event from targets: {targets}")
    return du, ev

dur_col, evt_col = pick_outcomes()

def event_array(s):
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce").fillna(0)
        vals = set(v.dropna().unique())
        return (v.astype(int) if vals <= {0, 1} else (v > 0).astype(int)).values
    t = s.astype(str).str.strip().str.lower()
    return (t.isin(["1", "1.0", "true", "dead", "deceased", "death"]) |
            t.str.contains(r"dead|deceased|death|1:", regex=True, na=False)).astype(int).values

dur = pd.to_numeric(val[dur_col], errors="coerce").values
evt = event_array(val[evt_col])

cands = [
    ("cox_baseline", "07_model_train_cox_AK6RYN4A/model.pkl"),
    ("alternative", "08_model_train_alternative_D78GTAGH/alternative_model.pkl"),
]
rows, cidx, models = [], {}, {}
for name, key in cands:
    m = unwrap(load(key))
    models[name] = m
    X = prepare_X(m, val)
    r = risk(m, X)
    ok = np.isfinite(dur) & np.isfinite(r)
    c = float(concordance_index(dur[ok], -r[ok], evt[ok]))
    cidx[name] = c
    rows.append((name, primary, c))
    mlca_runtime.record_metric(primary, c, "val")

base = models["cox_baseline"]
ph_est = final_est(base)
Xb = prepare_X(base, val)
if base is not ph_est and hasattr(base, "named_steps") and len(base.named_steps) > 1:
    try:
        Xb = base[:-1].transform(Xb)
    except Exception:
        pass
Xb = to_frame(Xb, val.index)
names = expected_names(ph_est)
if names and Xb.shape[1] == len(names):
    Xb.columns = names
if names:
    Xb = Xb[[c for c in names if c in Xb.columns]]
ph = Xb.copy()
ph[dur_col] = dur
ph[evt_col] = evt
try:
    res = proportional_hazard_test(ph_est, ph, time_transform="rank")
    pser = res.summary["p"] if "p" in res.summary.columns else getattr(res, "p_values", pd.Series(dtype=float))
    for cov, pv in pser.items():
        rows.append(("cox_baseline", f"ph_p_{cov}", float(pv)))
except Exception:
    rows.append(("cox_baseline", "ph_test_unavailable", np.nan))

metrics = pd.DataFrame(rows, columns=["model", "metric", "value"])
metrics["value"] = pd.to_numeric(metrics["value"], errors="coerce").map(
    lambda x: np.nan if pd.isna(x) else float(f"{x:.4g}")
)
if len(metrics) > 50:
    keep = metrics[metrics["metric"] == primary]
    rest = metrics[metrics["metric"] != primary].sort_values("value", na_position="last").head(max(0, 50 - len(keep)))
    metrics = pd.concat([keep, rest], ignore_index=True)
mlca_runtime.save_table(metrics, "validation_metrics.csv", "Validation C-index and baseline Schoenfeld p-values")

higher_better = primary.lower() not in {"loss", "mae", "mse", "rmse", "log_loss", "brier_score"}
best_name = (max if higher_better else min)(cidx, key=cidx.get)
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_c_index", float(cidx[best_name]), source="validation_metrics.csv")
print(f"train={train.shape} val={val.shape} candidates={len(cands)}")
print(f"best={best_name} c_index={cidx[best_name]:.4f} rows={len(metrics)}")
