import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in protocol:
        raise KeyError(f"Missing protocol key {k}; available: {list(protocol)}")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
task = protocol["task"]
primary = task["primary_metric"]
group_col = protocol.get("group_column")
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

event_col = task.get("event_column") or protocol.get("event_column")
duration_col = task.get("duration_column") or task.get("duration") or task.get("time_column") or protocol.get("duration_column")
if not event_col:
    ev = [c for c in targets if any(k in c.lower() for k in ("status", "event", "dead", "death"))]
    if ev:
        event_col = ev[0]
if not duration_col:
    du = [c for c in targets if c != event_col and any(k in c.lower() for k in ("month", "time", "duration", "day", "year"))]
    if not du:
        du = [c for c in targets if c != event_col]
    if du:
        duration_col = du[0]
if not event_col or not duration_col:
    raise KeyError(f"Cannot infer event/duration columns; available top-level keys: {list(protocol)}")

excl = set(targets) | set(excluded)
if group_col:
    excl.add(group_col)
feature_cols = []
for c in train.columns:
    if c in excl:
        continue
    s = train[c]
    if s.nunique(dropna=True) == len(s):
        nm = str(c).lower()
        if any(k in nm for k in ("id", "code", "key", "patient", "sample", "case", "index")) or pd.api.types.is_object_dtype(s):
            continue
    feature_cols.append(c)

def load_pkl(key):
    return pickle.loads(inputs[key].read_bytes())

preprocessor = load_pkl("05_preprocess_fit_preprocessing_QC6SFFG9/preprocessor.pkl")
selector = load_pkl("06_features_select_features_3MAAM9NK/selector.pkl")
cox_model = load_pkl("07_model_train_cox_07QBRYRX/model.pkl")
alt_model = load_pkl("latest_model")

def model_name(path_key, fallback):
    try:
        m = json.loads(inputs[path_key].read_text())
        for k in ("model_name", "name", "estimator_name", "algorithm"):
            if isinstance(m.get(k), str):
                return m[k]
        if isinstance(m.get("model"), dict):
            for k in ("name", "model_name"):
                if isinstance(m["model"].get(k), str):
                    return m["model"][k]
    except Exception:
        pass
    return fallback

def to_event(s):
    if pd.api.types.is_numeric_dtype(s):
        return (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype(int)
    lc = s.astype(str).str.lower()
    return lc.str.contains("dead|deceased|event|true|yes|1", regex=True).astype(int)

def apply_step(step, X, idx):
    if step == "passthrough" or not hasattr(step, "transform"):
        return X
    cols = list(getattr(step, "feature_names_in_", []) or [])
    Xt = step.transform(X[cols]) if isinstance(X, pd.DataFrame) and cols and all(c in X.columns for c in cols) else step.transform(X)
    if not isinstance(Xt, pd.DataFrame):
        out = list(getattr(step, "get_feature_names_out", lambda: [])())
        if len(out) == Xt.shape[1]:
            Xt = pd.DataFrame(Xt, index=idx, columns=out)
    return Xt

def build_input(df, model):
    cols = list(getattr(model, "feature_names_in_", []) or []) or list(getattr(model, "params", pd.Series(dtype=float)).index)
    if cols and all(c in df.columns for c in cols):
        return df[cols]
    X = df[feature_cols]
    if selector is not None and hasattr(selector, "transform"):
        X = apply_step(selector, X, df.index)
    elif preprocessor is not None:
        X = apply_step(preprocessor, X, df.index)
    if cols and isinstance(X, pd.DataFrame) and all(c in X.columns for c in cols):
        return X[cols]
    return X

def risk_scores(model, df):
    if hasattr(model, "predict_partial_hazard"):
        return np.asarray(model.predict_partial_hazard(build_input(df, model))).ravel()
    return np.asarray(model.predict(build_input(df, model))).ravel()

def find_cph(model):
    if hasattr(model, "predict_partial_hazard"):
        return model
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "predict_partial_hazard"):
                return step
    return None

def cph_frame(model, cph, df):
    if model is cph:
        return build_input(df, cph)
    X = df[feature_cols]
    for _, step in model.named_steps.items():
        if step is cph:
            break
        X = apply_step(step, X, df.index)
    if not isinstance(X, pd.DataFrame):
        cols = list(getattr(cph, "params", pd.Series(dtype=float)).index)
        X = pd.DataFrame(X, index=df.index, columns=cols if len(cols) == X.shape[1] else None)
    return X

def sig4(v):
    return float(f"{float(v):.4g}")

candidates = [(model_name("07_model_train_cox_07QBRYRX/manifest.json", "cox"), cox_model),
              (model_name("08_model_train_alternative_5GWPCP1V/manifest.json", "alternative"), alt_model)]
T = pd.to_numeric(val[duration_col], errors="coerce")
E = to_event(val[event_col])
rows, pairs = [], []
for name, model in candidates:
    risk = pd.Series(np.asarray(risk_scores(model, val)).ravel(), index=val.index)
    mask = T.notna() & np.isfinite(risk)
    cidx = float(concordance_index(T[mask], -risk[mask], E[mask]))
    rows.append([name, primary, sig4(cidx)])
    pairs.append((name, cidx))
    mlca_runtime.record_metric(primary, cidx, split="val")

baseline_name, baseline_model = candidates[0]
cph = find_cph(baseline_model)
if cph is not None:
    try:
        from lifelines.statistics import proportional_hazard_test
        ph = cph_frame(baseline_model, cph, val)
        ph[duration_col] = T
        ph[event_col] = E
        summ = proportional_hazard_test(cph, ph, time_transform="rank").summary
        for i, (idx, row) in enumerate(summ.iterrows()):
            if len(rows) >= 50:
                break
            cov = str(idx[0]) if isinstance(idx, tuple) else str(idx)
            rows.append([baseline_name, f"schoenfeld_p_{cov}", sig4(row["p"])])
    except Exception:
        pass

mlca_runtime.save_table(pd.DataFrame(rows, columns=["model", "metric", "value"]), "validation_metrics.csv", caption="Validation metrics")
lower_better = str(primary).lower() in {"rmse", "mae", "mse", "log_loss", "brier", "loss"}
best_name, best_val = (min(pairs, key=lambda x: x[1]) if lower_better else max(pairs, key=lambda x: x[1]))
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_c_index", sig4(best_val), source="validation_metrics.csv")
print(f"train={train.shape} val={val.shape} features={len(feature_cols)}")
print(f"best={best_name} {primary}={best_val:.4f}")
