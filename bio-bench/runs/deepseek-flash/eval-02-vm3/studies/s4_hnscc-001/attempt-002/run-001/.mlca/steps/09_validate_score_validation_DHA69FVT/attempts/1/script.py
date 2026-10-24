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

def load(key):
    with open(P[key], "rb") as f:
        return pickle.load(f)

def unwrap(o):
    if isinstance(o, dict):
        for k in ["model", "estimator", "pipeline"]:
            if k in o:
                return o[k]
    return o

pre = load("05_preprocess_fit_preprocessing_M6JRF55T/preprocessor.pkl")
sel = load("06_features_select_features_64YJBEKC/selector.pkl")

def expected_names(o):
    if o is None:
        return None
    for attr in ["feature_names_in_"]:
        if hasattr(o, attr):
            return list(getattr(o, attr))
    if hasattr(o, "params_"):
        return list(o.params_.index)
    return None

def transform_with(sel, pre, X):
    sn = expected_names(sel)
    if sel is not None and sn is not None and all(c in X.columns for c in sn):
        return sel.transform(X)
    Z = X
    if pre is not None:
        Z = pre.transform(Z)
    if sel is not None:
        Z = sel.transform(Z)
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
            return Z
        return pd.DataFrame(Z, columns=exp, index=df.index)
    return Z

def risk(model, X):
    if hasattr(model, "predict_partial_hazard"):
        return np.asarray(model.predict_partial_hazard(X)).ravel()
    if hasattr(model, "predict_log_partial_hazard"):
        return np.asarray(model.predict_log_partial_hazard(X)).ravel()
    if hasattr(model, "predict_expectation"):
        return -np.asarray(model.predict_expectation(X)).ravel()
    if hasattr(model, "predict_median"):
        return -np.asarray(model.predict_median(X)).ravel()
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.asarray(model.predict(X)).ravel()

def pick_outcomes():
    ev = next((c for c in targets if re.search(r"status|event|dead|death", c, re.I)), None)
    du = next((c for c in targets if c != ev and re.search(r"month|time|duration|survival", c, re.I)), None)
    if du is None:
        du = next(c for c in targets if c != ev)
    if ev is None:
        ev = next(c for c in targets if c != du)
    return du, ev

dur_col, evt_col = pick_outcomes()

def event_array(s):
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce").fillna(0)
        vals = set(v.dropna().unique())
        return (v.astype(int) if vals <= {0, 1} else (v > 0).astype(int)).values
    t = s.astype(str).str.lower()
    return t.str.contains(r"dead|deceased|death|1:", regex=True, na=False).astype(int).values

dur = pd.to_numeric(val[dur_col], errors="coerce").values
evt = event_array(val[evt_col])

cands = [
    ("cox_baseline", "07_model_train_cox_AK6RYN4A/model.pkl"),
    ("alternative", "08_model_train_alternative_D78GTAGH/alternative_model.pkl"),
]
rows, cidx = [], {}
for name, key in cands:
    m = unwrap(load(key))
    X = prepare_X(m, val)
    r = risk(m, X)
    ok = np.isfinite(dur) & np.isfinite(evt) & np.isfinite(r)
    c = float(concordance_index(dur[ok], -r[ok], evt[ok]))
    cidx[name] = c
    rows.append((name, primary, c))
    mlca_runtime.record_metric(primary, c, "val")

base = unwrap(load(cands[0][1]))
try:
    xcols = list(base.params_.index)
    Xb = prepare_X(base, val)
    if not isinstance(Xb, pd.DataFrame):
        Xb = pd.DataFrame(Xb, columns=xcols, index=val.index)
    ph = Xb.copy()
    ph[dur_col] = dur
    ph[evt_col] = evt
    res = proportional_hazard_test(base, ph, time_transform="rank")
    pser = res.summary["p"] if hasattr(res, "summary") and "p" in res.summary.columns else getattr(res, "p_values", pd.Series(dtype=float))
    for cov, pv in pser.items():
        rows.append(("cox_baseline", f"ph_p_{cov}", float(pv)))
except Exception:
    pass

metrics = pd.DataFrame(rows, columns=["model", "metric", "value"])
metrics["value"] = metrics["value"].map(lambda x: np.nan if pd.isna(x) else float(f"{x:.4g}"))
if len(metrics) > 50:
    keep_c = metrics[metrics["metric"] == primary]
    ph = metrics[metrics["metric"] != primary].sort_values("value", na_position="last").head(max(0, 48 - len(keep_c)))
    metrics = pd.concat([keep_c, ph], ignore_index=True)
mlca_runtime.save_table(metrics, "validation_metrics.csv", "Validation C-index and baseline Schoenfeld proportional-hazards p-values")

higher_better = primary.lower() not in {"loss", "mae", "mse", "rmse", "log_loss", "brier_score"}
best_name = (max if higher_better else min)(cidx, key=cidx.get)
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_c_index", float(cidx[best_name]), source="validation_metrics.csv")
print(f"train={train.shape} val={val.shape} candidates={len(cands)}")
print(f"best={best_name} c_index={cidx[best_name]:.4f} metric_rows={len(metrics)}")
