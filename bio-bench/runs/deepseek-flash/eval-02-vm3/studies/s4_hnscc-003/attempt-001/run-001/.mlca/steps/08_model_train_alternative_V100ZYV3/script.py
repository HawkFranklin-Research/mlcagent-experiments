import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
except KeyError as e:
    raise KeyError(f"Missing {e}; available top-level keys: {list(protocol)}") from e
group_col = protocol.get("group_column")
exclude_set = set(targets) | set(excluded)
if group_col:
    exclude_set.add(group_col)

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
pre = load_pkl(inp["05_preprocess_fit_preprocessing_RJBK5NR5/preprocessor.pkl"])
sel = load_pkl(inp["06_features_select_features_QN7Y384S/selector.pkl"])

event_col = next((c for c in targets if any(k in str(c).lower() for k in ("status", "event", "dead", "death"))), None)
if event_col is None:
    event_col = targets[-1] if len(targets) > 1 else targets[0]
duration_col = next((c for c in targets if c != event_col), targets[0])

def input_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

pre_names = input_names(pre)
if pre_names:
    feature_columns = [c for c in pre_names if c in train.columns and c not in exclude_set]
else:
    feature_columns = []
    n = len(train)
    for c in train.columns:
        if c in exclude_set:
            continue
        s = train[c]
        if pd.api.types.is_numeric_dtype(s):
            if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == s.notna().sum():
                continue
            feature_columns.append(c)
        elif pd.api.types.is_datetime64_any_dtype(s):
            feature_columns.append(c)
        elif s.nunique(dropna=True) <= 0.05 * n:
            feature_columns.append(c)

for obj in (pre, sel):
    try:
        obj.set_output(transform="pandas")
    except Exception:
        pass

includes_pre = False
if hasattr(sel, "named_steps") and sel.named_steps:
    pre_in = getattr(pre, "feature_names_in_", None)
    sel_in = getattr(sel, "feature_names_in_", None)
    if pre_in is not None and sel_in is not None and list(sel_in) == list(pre_in):
        includes_pre = True
    if not includes_pre:
        for name, step in sel.named_steps.items():
            if "preprocess" in name.lower() or "preproc" in name.lower():
                includes_pre = True
                break
            if pre_in is not None and hasattr(step, "feature_names_in_") and list(step.feature_names_in_) == list(pre_in):
                includes_pre = True
                break

frontend = sel if includes_pre else Pipeline([("preprocessor", pre), ("selector", sel)])

def transform_front(front, X):
    Z = front.transform(X)
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    if not isinstance(Z, pd.DataFrame):
        names = None
        try:
            names = front.get_feature_names_out()
        except Exception:
            names = None
        if names is not None and len(names) == Z.shape[1]:
            Z = pd.DataFrame(Z, index=X.index, columns=names)
        else:
            Z = pd.DataFrame(Z, index=X.index)
    return Z

def encode_event(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).to_numpy()
    m = s.astype(str).str.strip().str.lower()
    return m.isin(["1", "true", "dead", "died", "deceased", "event", "yes", "y"]).astype(int).to_numpy()

S = transform_front(frontend, train[feature_columns].copy())
X_fit = S.copy()
X_fit[duration_col] = pd.to_numeric(train[duration_col], errors="coerce").astype(float).to_numpy()
X_fit[event_col] = encode_event(train[event_col])

cox = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
cox = mlca_runtime.fit_record(cox, X_fit, partition="train", duration_col=duration_col, event_col=event_col)

try:
    S_val = transform_front(frontend, val[feature_columns].copy()).reindex(columns=S.columns)
    risk = cox.predict_partial_hazard(S_val).to_numpy().ravel()
    t = pd.to_numeric(val[duration_col], errors="coerce").astype(float).to_numpy()
    e = encode_event(val[event_col])
    ci = concordance_index(t, -risk, e)
    mlca_runtime.record_metric("c_index", float(ci), split="val")
    print("alternative val c_index", round(float(ci), 4))
except Exception as exc:
    print("alternative val c_index skipped", type(exc).__name__)

alt = {
    "model": cox,
    "cox": cox,
    "estimator": cox,
    "frontend": frontend,
    "preprocessor": pre,
    "selector": sel,
    "duration_col": duration_col,
    "event_col": event_col,
    "feature_columns": list(feature_columns),
    "model_columns": list(S.columns),
    "includes_preprocessing": bool(includes_pre),
    "penalizer": 0.1,
    "l1_ratio": 0.5,
}
mlca_runtime.save_model(alt, "alternative_model.pkl")
print("saved alternative_model.pkl")
