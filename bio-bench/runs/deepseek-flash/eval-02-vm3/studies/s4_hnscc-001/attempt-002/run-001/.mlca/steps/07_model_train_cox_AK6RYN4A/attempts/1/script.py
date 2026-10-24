import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from lifelines import CoxPHFitter

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns"):
    if key not in protocol:
        raise KeyError(f"missing {key}; available keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
train = pd.read_parquet(inputs["train"])

with open(inputs["05_preprocess_fit_preprocessing_M6JRF55T/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_64YJBEKC/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        try:
            return list(obj.feature_names_in_)
        except Exception:
            return None
    return None

pre_cols = input_cols(pre)
sel_cols = input_cols(sel)
pre_out = None
if hasattr(pre, "get_feature_names_out"):
    try:
        pre_out = [str(c) for c in pre.get_feature_names_out()]
    except Exception:
        pass

base_cols = [c for c in train.columns if c not in set(targets) | set(excluded) | ({group} if group else set())]
def safe_cols(cols):
    if cols is None:
        return base_cols
    bad = set(cols) & (set(targets) | set(excluded) | ({group} if group else set()))
    if bad:
        raise ValueError(f"columns reach excluded data: {sorted(bad)}")
    missing = [c for c in cols if c not in train.columns]
    if missing:
        raise KeyError(f"missing expected columns: {missing}")
    return list(cols)

chain_pre = True
if sel_cols is not None:
    if pre_out is not None and set(sel_cols).issubset(set(pre_out)):
        chain_pre = True
    elif pre_cols is not None and set(sel_cols).issubset(set(pre_cols)):
        chain_pre = False
    elif isinstance(sel, Pipeline):
        chain_pre = False

pre_in = safe_cols(pre_cols) if chain_pre else None
sel_in = safe_cols(sel_cols) if not chain_pre else None

def transform_features(df):
    if chain_pre:
        X = pre.transform(df[pre_in])
        if not isinstance(X, pd.DataFrame):
            names = pre_out or [f"f{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, index=df.index, columns=names)
        else:
            X.index = df.index
        if sel_cols is not None and all(c in X.columns for c in sel_cols):
            X = X[sel_cols]
        X = sel.transform(X)
    else:
        X = sel.transform(df[sel_in])
    if not isinstance(X, pd.DataFrame):
        names = None
        if hasattr(sel, "get_feature_names_out"):
            try:
                names = [str(c) for c in sel.get_feature_names_out()]
            except Exception:
                pass
        if names is None:
            names = [f"f{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, index=df.index, columns=names)
    else:
        X.index = df.index
    return X

def to_event(s):
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        vals = set(s.dropna().unique())
        if vals <= {0, 1}:
            return s.astype(int)
        return (s > 0).astype(int)
    mapping = {}
    for v in s.dropna().unique():
        lv = str(v).strip().lower()
        mapping[v] = 1 if lv in {"dead", "deceased", "1", "true", "yes", "event", "died"} else 0
    return s.map(mapping).astype(int)

duration_col = None
event_col = None
for c in targets:
    name = str(c).lower()
    if any(k in name for k in ("month", "time", "duration", "survival")) and not any(k in name for k in ("status", "event")):
        duration_col = c
    elif any(k in name for k in ("status", "event", "dead", "alive", "censor")):
        event_col = c
if duration_col is None or event_col is None:
    for c in targets:
        if train[c].nunique(dropna=True) <= 2:
            event_col = event_col or c
        else:
            duration_col = duration_col or c
if duration_col is None or event_col is None or duration_col == event_col:
    raise ValueError(f"could not resolve duration/event columns from targets {targets}")

fit_mask = train[duration_col].notna() & train[event_col].notna()
train_fit = train.loc[fit_mask].copy()
X_feat = transform_features(train_fit)

X_fit = X_feat.copy()
X_fit[duration_col] = train_fit[duration_col].astype(float)
X_fit[event_col] = to_event(train_fit[event_col])

cph = CoxPHFitter(penalizer=0.0)
cph = mlca_runtime.fit_record(
    cph,
    X_fit,
    "train",
    duration_col=duration_col,
    event_col=event_col,
)

model = {
    "preprocessor": pre if chain_pre else None,
    "selector": sel,
    "cox": cph,
    "chain_pre": chain_pre,
    "input_columns": pre_in if chain_pre else sel_in,
    "preprocessor_input_columns": pre_in,
    "selector_input_columns": sel_cols,
    "duration_col": duration_col,
    "event_col": event_col,
}
mlca_runtime.save_model(model, "model.pkl")
mlca_runtime.record_metric("train_c_index", float(cph.concordance_index_), "train")
print(f"train rows: {len(train_fit)}; transformed features: {X_feat.shape[1]}")
print(f"duration={duration_col}; event={event_col}; chain_pre={chain_pre}")
print(f"train c-index: {cph.concordance_index_:.4f}")
