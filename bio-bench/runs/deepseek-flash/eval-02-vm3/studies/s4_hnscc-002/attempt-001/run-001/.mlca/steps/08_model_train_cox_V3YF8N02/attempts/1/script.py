import mlca_runtime, json, pickle
import pandas as pd
from lifelines import CoxPHFitter

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

def need(d, k, where):
    if k not in d:
        raise KeyError(f"Missing {k!r} in {where}; available: {sorted(d)}")
    return d[k]

targets = need(protocol, "targets", "protocol")
excluded = set(need(protocol, "excluded_columns", "protocol"))
group_col = protocol.get("group_column")
task = need(protocol, "task", "protocol")
primary_metric = need(task, "primary_metric", "task")
duration_col = (
    protocol.get("duration_column")
    or task.get("duration_column")
    or protocol.get("time_column")
    or task.get("time_column")
    or "Overall Survival (Months)"
)
if len(targets) != 1:
    raise ValueError("Expected exactly one event target")
event_col = targets[0]

train = pd.read_parquet(inp["train"])
for c in (duration_col, event_col):
    if c not in train.columns:
        raise KeyError(f"{c!r} not in train; columns: {list(train.columns)}")

def load_pkl(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("06_preprocess_fit_preprocessing_GDNFB5M3/preprocessor.pkl")
sel = load_pkl("07_features_select_features_JN73C6P9/selector.pkl")

outcome = set(targets) | {duration_col}
feat_cols = [
    c for c in train.columns
    if c not in outcome and c not in excluded and c != group_col
]
key_like = []
for c in feat_cols:
    s = train[c]
    is_text = (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )
    if is_text and len(s) > 1 and s.nunique(dropna=True) == len(s):
        key_like.append(c)
feat_cols = [c for c in feat_cols if c not in key_like]

pre_names = list(getattr(pre, "feature_names_in_", [])) or feat_cols
for bad in outcome | excluded | ({group_col} if group_col else set()):
    if bad in pre_names:
        raise ValueError(f"Outcome/excluded column {bad!r} in preprocessor input")

sel_names = list(getattr(sel, "feature_names_in_", [])) if sel is not None else []
use_sel_only = bool(
    sel is not None
    and sel_names
    and all(n in pre_names for n in sel_names)
)
if not use_sel_only and hasattr(sel, "named_steps"):
    for step in sel.named_steps.values():
        if hasattr(step, "feature_names_in_") and set(step.feature_names_in_) == set(pre_names):
            use_sel_only = True
            break

def transform(df):
    if use_sel_only:
        Z = sel.transform(df[sel_names])
    else:
        Z = pre.transform(df[pre_names])
        if sel is not None:
            Z = sel.transform(Z)
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    if isinstance(Z, pd.DataFrame):
        Z = Z.copy()
        Z.index = df.index
        return Z
    try:
        names = (sel if use_sel_only else pre).get_feature_names_out()
        if sel is not None and not use_sel_only:
            names = None
    except Exception:
        names = None
    if names is None or len(names) != Z.shape[1]:
        names = [f"f{i}" for i in range(Z.shape[1])]
    return pd.DataFrame(Z, index=df.index, columns=names)

X = transform(train)
X_fit = X.copy()
X_fit[duration_col] = pd.to_numeric(train[duration_col], errors="raise").to_numpy()

def encode_event(s):
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    t = s.astype(str).str.lower()
    pos = t.isin({"1", "true", "yes", "dead", "deceased", "event"})
    if pos.nunique() < 2 and t.nunique() > 1:
        raise ValueError(f"Cannot infer event coding from values {sorted(t.unique())[:5]}")
    return pos.astype(int)

X_fit[event_col] = encode_event(train[event_col]).to_numpy()

cox = CoxPHFitter(penalizer=0.0)
mlca_runtime.fit_record(
    cox, X_fit, "train",
    duration_col=duration_col,
    event_col=event_col,
)

payload = {
    "preprocessor": pre,
    "selector": sel,
    "cox": cox,
    "use_selector_only": use_sel_only,
    "pre_names": pre_names,
    "sel_names": sel_names,
    "duration_col": duration_col,
    "event_col": event_col,
    "feature_columns": feat_cols,
}
mlca_runtime.save_model(payload, "model.pkl")
print(f"train rows={len(train)} features={len(pre_names)} transformed={X.shape[1]}")
if hasattr(cox, "concordance_index_"):
    print(f"train c_index={cox.concordance_index_:.4f}")
