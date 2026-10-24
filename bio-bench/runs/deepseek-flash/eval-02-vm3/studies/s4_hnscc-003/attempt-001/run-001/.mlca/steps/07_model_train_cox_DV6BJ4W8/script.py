import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from lifelines import CoxPHFitter

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
train = pd.read_parquet(paths["train"])

targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
drop = set(targets) | set(excluded) | ({group} if group else set())

DURATION, EVENT = "Overall Survival (Months)", "Overall Survival Status"
duration_col, event_col = DURATION, EVENT
if duration_col not in targets or event_col not in targets:
    numeric_targets = [c for c in targets if pd.api.types.is_numeric_dtype(train[c])]
    duration_col = max(numeric_targets, key=lambda c: train[c].nunique())
    event_col = next(c for c in targets if c != duration_col and train[c].nunique() <= 2)

raw_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and \
       s.nunique(dropna=False) == len(train) and len(train) > 1:
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(train) and len(train) > 1:
        if s.is_monotonic_increasing or "id" in str(c).lower():
            continue
    raw_cols.append(c)

with open(paths["05_preprocess_fit_preprocessing_RJBK5NR5/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(paths["06_features_select_features_QN7Y384S/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_cols(obj):
    try:
        cols = getattr(obj, "feature_names_in_", None)
        if cols is not None:
            return list(cols)
    except Exception:
        pass
    if isinstance(obj, Pipeline):
        for st in obj.named_steps.values():
            cols = input_cols(st)
            if cols is not None:
                return cols
    return None

pre_in, sel_in = input_cols(pre), input_cols(sel)
def valid(cols):
    return cols is not None and all(c in train.columns for c in cols) and not any(c in drop for c in cols)

cands = []
if valid(sel_in) and isinstance(sel, Pipeline) and len(sel.named_steps) >= 2:
    cands.append((sel, sel_in))
if valid(pre_in):
    cands.append((Pipeline([("preprocessor", pre), ("selector", sel)]), pre_in))
if valid(sel_in):
    cands.append((sel, sel_in))
if not cands:
    cands.append((Pipeline([("preprocessor", pre), ("selector", sel)]), raw_cols))

frontend, Xt = None, None
for f, cols in cands:
    try:
        Xt = f.transform(train[cols])
        if Xt is not None:
            frontend, raw_cols = f, list(cols)
            break
    except Exception:
        continue
if frontend is None:
    raise RuntimeError("No compatible preprocessing/selector frontend could transform the training data")

if isinstance(Xt, pd.DataFrame):
    X_feat = Xt.copy()
    X_feat.index = train.index
else:
    try:
        names = frontend.get_feature_names_out()
    except Exception:
        names = [f"x{i}" for i in range(Xt.shape[1])]
    X_feat = pd.DataFrame(Xt, columns=names, index=train.index)
X_feat.columns = [str(c) for c in X_feat.columns]
if X_feat.columns.duplicated().any():
    X_feat.columns = [f"{c}_{i}" for i, c in enumerate(X_feat.columns)]

X_fit = pd.concat([X_feat, train[[duration_col, event_col]].copy()], axis=1)
X_fit[duration_col] = pd.to_numeric(X_fit[duration_col], errors="coerce")
if not pd.api.types.is_numeric_dtype(X_fit[event_col]):
    ev = X_fit[event_col].astype(str).str.lower()
    X_fit[event_col] = ev.isin(["dead", "deceased", "1", "true", "yes", "event"]).astype(int)

cph = CoxPHFitter(penalizer=0.0)
cph = mlca_runtime.fit_record(
    cph, X_fit, "train",
    duration_col=duration_col, event_col=event_col, show_progress=False
)

bundle = {
    "frontend": frontend,
    "cph": cph,
    "feature_columns": list(raw_cols),
    "duration_col": duration_col,
    "event_col": event_col,
    "model_feature_columns": list(X_feat.columns),
}
mlca_runtime.save_model(bundle, "model.pkl")

if hasattr(cph, "concordance_index_"):
    mlca_runtime.record_metric("train_c_index", float(cph.concordance_index_), split="train")
    print(f"train c-index={cph.concordance_index_:.4f}")
print(f"train rows={len(train)}; raw features={len(raw_cols)}; model features={X_feat.shape[1]}")
print(f"duration={duration_col}; event={event_col}")
