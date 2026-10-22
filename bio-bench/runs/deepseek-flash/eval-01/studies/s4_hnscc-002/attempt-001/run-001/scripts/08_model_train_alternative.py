import mlca_runtime, json, pickle
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
train = pd.read_parquet(inp["train"])

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
duration_col = "Overall Survival (Months)"
event_col = "Overall Survival Status"
if duration_col not in targets or event_col not in targets:
    raise ValueError(f"Expected survival targets in {targets}")

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load(inp["05_preprocess_fit_preprocessing_QC6SFFG9/preprocessor.pkl"])
sel = load(inp["06_features_select_features_3MAAM9NK/selector.pkl"])
pre_cols = list(getattr(pre, "feature_names_in_", []))
sel_cols = list(getattr(sel, "feature_names_in_", []))
if pre_cols and sel_cols and pre_cols == sel_cols:
    feature_pipe = sel
else:
    feature_pipe = Pipeline([("preprocessor", pre), ("selector", sel)])

def keylike(s):
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        return s.nunique(dropna=True) > 0.05 * len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.is_unique and s.nunique(dropna=True) > 0.9 * len(s)
    return False

drop = set(targets + excluded + ([group] if group else []))
derived = [c for c in train.columns if c not in drop]
derived = [c for c in derived if not keylike(train[c])]
cols = pre_cols if pre_cols else derived
cols = [c for c in cols if c in train.columns and c not in drop]

X_feat = train[cols]
Z = feature_pipe.transform(X_feat)
if isinstance(Z, pd.DataFrame):
    Zdf = Z.copy()
else:
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    try:
        names = feature_pipe.get_feature_names_out()
    except Exception:
        names = [f"x{i}" for i in range(Z.shape[1])]
    Zdf = pd.DataFrame(Z, index=train.index, columns=names)

time = pd.to_numeric(train[duration_col], errors="coerce")
ev = train[event_col]
if pd.api.types.is_numeric_dtype(ev):
    ev = pd.to_numeric(ev, errors="coerce")
else:
    ev = ev.astype(str).str.strip().str.lower().map(
        {"dead": 1, "deceased": 1, "alive": 0, "living": 0, "1": 1, "0": 0}
    )

X = Zdf.copy()
X[duration_col] = time
X[event_col] = ev
X = X.loc[X[[duration_col, event_col]].notna().all(axis=1)]
print("features", len(cols), "model matrix", X.shape)

cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
fitted = mlca_runtime.fit_record(
    cph, X, "train", duration_col=duration_col, event_col=event_col
)

try:
    baseline = load(inp["07_model_train_cox_07QBRYRX/model.pkl"])
except Exception:
    baseline = None

if isinstance(baseline, dict):
    obj = dict(baseline)
    for k, v in list(obj.items()):
        if isinstance(v, CoxPHFitter):
            obj[k] = fitted
    obj.setdefault("cox", fitted)
    obj.setdefault("feature_pipeline", feature_pipe)
    obj.setdefault("feature_columns", cols)
    obj.setdefault("duration_col", duration_col)
    obj.setdefault("event_col", event_col)
else:
    obj = fitted

mlca_runtime.save_model(obj, "alternative_model.pkl")
print("saved alternative_model.pkl")
