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

def need(d, k, ctx):
    if k not in d:
        raise KeyError(f"Missing {k} in {ctx}; available keys: {list(d.keys())}")
    return d[k]

targets = need(protocol, "targets", "protocol")
excluded = need(protocol, "excluded_columns", "protocol")
group_col = need(protocol, "group_column", "protocol")
task = need(protocol, "task", "protocol")
if not isinstance(targets, list) or not targets:
    raise ValueError("protocol['targets'] must be a non-empty list")
duration_col = targets[0]

event_col = None
for container in (task, protocol):
    for key in ("event_column", "event_col", "event"):
        if key in container and isinstance(container[key], str):
            event_col = container[key]
            break
    if event_col:
        break
if event_col is None:
    raise KeyError(f"Cannot find event column; task keys={list(task.keys())}, protocol keys={list(protocol.keys())}")

def is_text_key(s):
    return (
        (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s))
        and s.nunique(dropna=True) == s.notna().sum()
        and len(s) > 1
    )

candidate = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group_col and not is_text_key(train[c])
]
if hasattr(pre, "feature_names_in_"):
    pre_cols = list(pre.feature_names_in_)
    bad = [c for c in pre_cols if c not in candidate]
    if bad:
        raise ValueError(f"Preprocessor expects columns outside feature set: {bad[:5]}")
    feature_cols = pre_cols
else:
    feature_cols = candidate

X_train = train[feature_cols].copy()
Xt = pre.transform(X_train)
if isinstance(Xt, pd.DataFrame):
    Xp = Xt
    out_cols = list(Xp.columns)
else:
    if hasattr(pre, "get_feature_names_out"):
        out_cols = list(pre.get_feature_names_out(feature_cols))
    else:
        out_cols = [f"f{i}" for i in range(Xt.shape[1])]
    if hasattr(Xt, "toarray"):
        Xp = pd.DataFrame(Xt.toarray(), columns=out_cols, index=X_train.index)
    else:
        Xp = pd.DataFrame(np.asarray(Xt), columns=out_cols, index=X_train.index)

def event_numeric(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all():
        return num.astype(float)
    lower = s.astype(str).str.lower()
    ev = (lower.str.contains("dead|deceased|death|event") | lower.str.match(r"^\s*1")).astype(float)
    if ev.sum() == 0 or ev.sum() == len(ev):
        cats = sorted(s.dropna().astype(str).unique())
        mapping = {v: i for i, v in enumerate(cats)}
        ev = s.astype(str).map(mapping).astype(float)
    return ev

ev = event_numeric(train[event_col])
dur = pd.to_numeric(train[duration_col], errors="coerce")
valid = dur.notna() & ev.notna()
pvals = {}
for col in out_cols:
    s = Xp[col]
    if s.nunique(dropna=True) <= 1:
        continue
    df = pd.DataFrame({duration_col: dur[valid], event_col: ev[valid], col: s[valid]}).dropna()
    if len(df) < 10 or df[event_col].sum() == 0:
        continue
    try:
        cph = CoxPHFitter()
        cph.fit(df, duration_col=duration_col, event_col=event_col)
        p = float(cph.summary.loc[col, "p"])
        if np.isfinite(p):
            pvals[col] = p
    except Exception:
        continue

selected = [c for c, p in sorted(pvals.items(), key=lambda kv: kv[1]) if p < 0.10][:20]

if not selected:
    if len(out_cols) <= 300:
        sel = ColumnTransformer([("drop", "drop", out_cols)], remainder="drop")
    else:
        sel = ColumnTransformer([("drop", "drop", list(range(len(out_cols))))], remainder="drop")
    pipe = Pipeline([("preprocess", pre), ("select", sel)])
elif len(out_cols) <= 300:
    if hasattr(pre, "set_output"):
        try:
            pre.set_output(transform="pandas")
        except Exception:
            pass
    sel = ColumnTransformer([("keep", "passthrough", selected)], remainder="drop")
    if hasattr(sel, "set_output"):
        sel.set_output(transform="pandas")
    pipe = Pipeline([("preprocess", pre), ("select", sel)])
    if hasattr(pipe, "set_output"):
        pipe.set_output(transform="pandas")
else:
    idx = [out_cols.index(c) for c in selected]
    sel = ColumnTransformer([("keep", "passthrough", idx)], remainder="drop")
    pipe = Pipeline([("preprocess", pre), ("select", sel)])

mlca_runtime.fit_record(pipe, X_train, partition="train")
mlca_runtime.save_model(pipe, "selector.pkl")
mlca_runtime.record_metric("n_selected_features", float(len(selected)), "train")
print(f"features={len(feature_cols)} processed={len(out_cols)} selected={len(selected)}")
