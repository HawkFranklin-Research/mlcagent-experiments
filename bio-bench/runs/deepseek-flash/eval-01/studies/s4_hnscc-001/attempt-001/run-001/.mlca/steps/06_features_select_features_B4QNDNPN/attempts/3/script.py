import json, pickle
import numpy as np, pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from lifelines import CoxPHFitter

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
train = pd.read_parquet(inp["train"])
pre_obj = pickle.loads(inp["05_preprocess_fit_preprocessing_49B478YE/preprocessor.pkl"].read_bytes())
pre = pre_obj.get("preprocessor", pre_obj) if isinstance(pre_obj, dict) else pre_obj

targets = protocol["targets"]; excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column"); task = protocol["task"]
duration_col = targets[0]

def _find_event():
    for d in (task, protocol):
        if not isinstance(d, dict): continue
        for k in ("event_column", "event_col", "event", "events", "event_columns", "status_column"):
            v = d.get(k)
            if isinstance(v, str) and v in train.columns: return v
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, str) and x in train.columns: return x
    for d in (task, protocol):
        if not isinstance(d, dict): continue
        for k, v in d.items():
            if "event" in k.lower() or "status" in k.lower():
                if isinstance(v, str) and v in train.columns: return v
                if isinstance(v, list):
                    for x in v:
                        if isinstance(x, str) and x in train.columns: return x
    for cand in ["Overall Survival Status", "overall_survival_status", "OS_STATUS", "vital_status", "status"]:
        if cand in train.columns: return cand
        for c in train.columns:
            if c.lower() == cand.lower(): return c
    return None

event_col = _find_event()
if not event_col:
    raise KeyError(f"event column not found; task keys={list(task)}; protocol keys={list(protocol)}")

def is_text_key(s):
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == s.notna().sum() and len(s) > 1

names_in = list(pre.feature_names_in_) if hasattr(pre, "feature_names_in_") else None
if names_in:
    feature_cols = [c for c in names_in if c in train.columns]
else:
    feature_cols = [c for c in train.columns
                    if c not in targets and c not in excluded and c != group_col
                    and c != event_col and not is_text_key(train[c])]

X_train = train[feature_cols].copy()
try: pre.set_output(transform="pandas")
except Exception: pass
Xt = pre.transform(X_train)
if isinstance(Xt, pd.DataFrame):
    names = list(Xt.columns)
elif hasattr(pre, "get_feature_names_out"):
    names = list(pre.get_feature_names_out(feature_cols))
else:
    names = [f"f{i}" for i in range(Xt.shape[1])]

def get_col(j):
    if isinstance(Xt, pd.DataFrame): return Xt.iloc[:, j]
    if hasattr(Xt, "tocsc"):
        return pd.Series(np.asarray(Xt.tocsc()[:, j].todense()).ravel(), index=X_train.index)
    return pd.Series(np.asarray(Xt)[:, j], index=X_train.index)

def event_numeric(s):
    if pd.api.types.is_numeric_dtype(s): return s.astype(float).values
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all(): return num.astype(float).values
    low = s.astype(str).str.lower()
    ev = (low.str.contains("dead|deceased|death|event") | low.str.match(r"^\s*1")).astype(float)
    if ev.sum() == 0 or ev.sum() == len(ev):
        cats = sorted(s.dropna().astype(str).unique()); m = {v: i for i, v in enumerate(cats)}
        ev = s.astype(str).map(m).astype(float)
    return ev.values

leak_pats = [str(event_col).lower()] + [str(t).lower() for t in targets]
def is_leaky(n):
    nl = str(n).lower()
    return any(p in nl for p in leak_pats)

ev = event_numeric(train[event_col])
dur = pd.to_numeric(train[duration_col], errors="coerce").values
valid = pd.Series(~(np.isnan(dur) | pd.isna(ev)), index=X_train.index)
dur_v = dur[valid.values]; ev_v = ev[valid.values]

pvals = {}
for j, name in enumerate(names):
    if is_leaky(name): continue
    s = get_col(j).loc[valid]
    if s.nunique(dropna=True) <= 1: continue
    df = pd.DataFrame({duration_col: dur_v, event_col: ev_v, "x": s.values}).dropna()
    if len(df) < 20 or df[event_col].sum() == 0: continue
    try:
        cph = mlca_runtime.fit_record(CoxPHFitter(), df, "train", y=None,
                                      duration_col=duration_col, event_col=event_col)
        p = float(cph.summary.loc["x", "p"])
        if np.isfinite(p): pvals[name] = p
    except Exception:
        continue

selected_names = [n for n, p in sorted(pvals.items(), key=lambda kv: kv[1]) if p < 0.10][:20]
if selected_names:
    if isinstance(Xt, pd.DataFrame) and len(names) <= 300:
        sel = ColumnTransformer([("keep", "passthrough", selected_names)], remainder="drop")
    else:
        idx = [names.index(n) for n in selected_names]
        sel = ColumnTransformer([("keep", "passthrough", idx)], remainder="drop")
    pipe = Pipeline([("preprocess", pre), ("select", sel)])
else:
    pipe = Pipeline([("preprocess", pre)])

mlca_runtime.fit_record(pipe, X_train, partition="train")
mlca_runtime.save_model(pipe, "selector.pkl")
mlca_runtime.record_metric("n_selected_features", float(len(selected_names)), "train")
print(f"event={event_col} features={len(feature_cols)} processed={len(names)} selected={len(selected_names)}")
