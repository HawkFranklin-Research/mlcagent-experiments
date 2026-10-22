import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from lifelines import CoxPHFitter

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing {key}; available keys: {sorted(d.keys())}")
    return d[key]

targets = need(protocol, "targets")
excluded = set(need(protocol, "excluded_columns"))
group = protocol.get("group_column")
task = need(protocol, "task")

train = pd.read_parquet(inputs["train"])
preprocessor = pickle.loads(inputs["05_preprocess_fit_preprocessing_QC6SFFG9/preprocessor.pkl"].read_bytes())
selector = pickle.loads(inputs["06_features_select_features_3MAAM9NK/selector.pkl"].read_bytes())

event_candidates = [c for c in targets if any(k in c.lower() for k in ["status", "event", "dead", "deceased"])]
event_col = event_candidates[0] if event_candidates else targets[0]
duration_col = next(c for c in targets if c != event_col)

drop_cols = set(targets) | excluded
if group:
    drop_cols.add(group)
feature_cols = []
for c in train.columns:
    if c in drop_cols:
        continue
    s = train[c]
    key_like = (pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or pd.api.types.is_integer_dtype(s.dtype)) and s.nunique(dropna=True) == len(train)
    if not key_like:
        feature_cols.append(c)

def to_event(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    t = s.astype(str).str.lower().str.strip()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[t.str.contains("dead|deceased", regex=True)] = 1.0
    out[t.str.contains("alive|living", regex=True)] = 0.0
    out[t.isin(["1", "1.0"])] = 1.0
    out[t.isin(["0", "0.0"])] = 0.0
    return out

def apply_saved(obj, df, fallback_cols=None):
    if hasattr(obj, "feature_names_in_"):
        cols = list(obj.feature_names_in_)
        X = df.loc[:, cols].copy()
    elif fallback_cols is not None:
        X = df.loc[:, [c for c in fallback_cols if c in df.columns]].copy()
    else:
        X = df.copy()
    out = obj.transform(X) if hasattr(obj, "transform") else obj.predict(X)
    if isinstance(out, pd.DataFrame):
        return out
    if hasattr(out, "toarray"):
        out = out.toarray()
    names = None
    if hasattr(obj, "get_feature_names_out"):
        try:
            names = list(obj.get_feature_names_out())
        except Exception:
            names = None
    if names is None or len(names) != out.shape[1]:
        names = [f"x{i}" for i in range(out.shape[1])]
    return pd.DataFrame(out, index=X.index, columns=names)

pre_cols = getattr(preprocessor, "feature_names_in_", None)
sel_cols = getattr(selector, "feature_names_in_", None)
selector_only = sel_cols is not None and pre_cols is not None and list(sel_cols) == list(pre_cols)

if selector_only:
    X_features = apply_saved(selector, train, fallback_cols=feature_cols)
    raw_feature_cols = list(sel_cols)
    frontend = "selector_only"
else:
    X_features = apply_saved(preprocessor, train, fallback_cols=feature_cols)
    X_features = apply_saved(selector, X_features)
    raw_feature_cols = list(pre_cols) if pre_cols is not None else feature_cols
    frontend = "preprocessor_then_selector"

X_features = X_features.astype(float)
df_fit = X_features.copy()
df_fit[duration_col] = pd.to_numeric(train[duration_col], errors="coerce").astype(float)
df_fit[event_col] = to_event(train[event_col]).astype(float)

cph = CoxPHFitter(penalizer=0.0)
fitted_cph = mlca_runtime.fit_record(
    cph, df_fit, partition="train",
    duration_col=duration_col, event_col=event_col
)

model_bundle = {
    "preprocessor": preprocessor,
    "selector": selector,
    "cph": fitted_cph,
    "frontend": frontend,
    "feature_columns": raw_feature_cols,
    "duration_col": duration_col,
    "event_col": event_col,
}
mlca_runtime.save_model(model_bundle, "model.pkl")

print(f"train rows={len(train)}, raw features={len(feature_cols)}")
print(f"model features={X_features.shape[1]}, cph fitted")
print("saved model.pkl")
