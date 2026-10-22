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

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
task = protocol["task"]
duration_col = targets[0]

event_col = None
for c in (task, protocol):
    for k in ("event_column", "event_col", "event"):
        v = c.get(k) if hasattr(c, "get") else None
        if isinstance(v, str) and v in train.columns:
            event_col = v; break
    if event_col: break
if not event_col:
    raise KeyError(f"event column not found; task={list(task)}; protocol={list(protocol)}")

def is_text_key(s):
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == s.notna().sum() and len(s) > 1

cand = [c for c in train.columns
        if c not in targets and c not in excluded and c != group_col and not is_text_key(train[c])]
if hasattr(pre, "feature_names_in_"):
    feature_cols = [c for c in pre.feature_names_in_ if c in cand] or cand
else:
    feature_cols = cand

X_train = train[feature_cols].copy()
try:
    pre.set_output(transform="pandas")
except Exception:
    pass
Xt = pre.transform(X_train)

if isinstance(Xt, pd.DataFrame):
    names = list(Xt.columns)
elif hasattr(pre, "get_feature_names_out"):
    names = list(pre.get_feature_names_out(feature_cols))
else:
    names = [f"f{i}" for i in range(Xt.shape[1])]

def get_col(j):
    if isinstance(Xt, pd.DataFrame):
        return Xt.iloc[:, j]
    if hasattr(Xt, "tocsc"):
        col = Xt.tocsc()[:, j]
        return pd.Series(np.asarray(col.todense()).ravel(), index=X_train.index)
    return pd.Series(np.asarray(Xt)[:, j], index=X_train.index)

def event_numeric(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float).values
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all():
        return num.astype(float).values
    low = s.astype(str).str.lower()
    ev = (low.str.contains("dead|deceased|death|event") | low.str.match(r"^\s*1")).astype(float)
    if ev.sum() == 0 or ev.sum() == len(ev):
        cats = sorted(s.dropna().astype(str).unique())
        m = {v: i for i, v in enumerate(cats)}
        ev = s.astype(str).map(m).astype(float)
    return ev.values

ev = event_numeric(train[event_col])
dur = pd.to_numeric(train[duration_col], errors="coerce").values
valid = pd.Series(~(np.isnan(dur) | pd.isna(ev)), index=X_train.index)
dur_v = dur[valid.values]
ev_v = ev[valid.values]

pvals = {}
for j, name in enumerate(names):
    s = get_col(j).loc[valid]
    if s.nunique(dropna=True) <= 1:
        continue
    df = pd.DataFrame({duration_col: dur_v, event_col: ev_v, "x": s.values}).dropna()
    if len(df) < 20 or df[event_col].sum() == 0:
        continue
    try:
        cph = mlca_runtime.fit_record(CoxPHFitter(), df, "train", y=None,
                                      duration_col=duration_col, event_col=event_col)
        p = float(cph.summary.loc["x", "p"])
        if np.isfinite(p):
            pvals[name] = p
    except Exception:
        continue

selected_names = [n for n, p in sorted(pvals.items(), key=lambda kv: kv[1]) if p < 0.10][:20]
selected_idx = [names.index(n) for n in selected_names]

if isinstance(Xt, pd.DataFrame) and len(names) <= 300:
    spec = [("keep", "passthrough", selected_names)] if selected_names else [("drop", "drop", names)]
else:
    spec = [("keep", "passthrough", selected_idx)] if selected_idx else [("drop", "drop", list(range(len(names))))]

sel = ColumnTransformer(spec, remainder="drop")
pipe = Pipeline([("preprocess", pre), ("select", sel)])
try:
    sel.set_output(transform="pandas"); pipe.set_output(transform="pandas")
except Exception:
    pass
mlca_runtime.fit_record(pipe, X_train, partition="train")
mlca_runtime.save_model(pipe, "selector.pkl")
mlca_runtime.record_metric("n_selected_features", float(len(selected_names)), "train")
print(f"features={len(feature_cols)} processed={len(names)} selected={len(selected_names)}")
