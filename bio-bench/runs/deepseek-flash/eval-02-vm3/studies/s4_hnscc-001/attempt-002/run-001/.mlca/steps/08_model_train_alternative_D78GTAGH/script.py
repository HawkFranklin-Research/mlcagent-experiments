import mlca_runtime
import json, pickle
import pandas as pd
from lifelines import CoxPHFitter

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())

targets = list(proto['targets'])
excluded = list(proto.get('excluded_columns', []))
group_col = proto.get('group_column')
dur_col, evt_col = targets[0], targets[1]

train = pd.read_parquet(inp['train'])
pre = pickle.load(open(inp['05_preprocess_fit_preprocessing_M6JRF55T/preprocessor.pkl'], 'rb'))
sel = pickle.load(open(inp['06_features_select_features_64YJBEKC/selector.pkl'], 'rb'))

pre_in = getattr(pre, 'feature_names_in_', None)
sel_in = getattr(sel, 'feature_names_in_', None)
drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
raw_cols = list(pre_in) if pre_in is not None else [c for c in train.columns if c not in drop]
X_raw = train[raw_cols]

if sel_in is not None and pre_in is not None and set(sel_in) == set(pre_in):
    X_sel = sel.transform(X_raw)
else:
    X_pre = pre.transform(X_raw)
    if not isinstance(X_pre, pd.DataFrame):
        X_pre = pd.DataFrame(X_pre, index=X_raw.index)
    if sel_in is not None and all(c in X_pre.columns for c in sel_in):
        X_sel = sel.transform(X_pre[list(sel_in)])
    elif sel_in is not None and all(c in X_raw.columns for c in sel_in):
        X_sel = sel.transform(X_raw[list(sel_in)])
    else:
        X_sel = sel.transform(X_pre)

if not isinstance(X_sel, pd.DataFrame):
    X_sel = pd.DataFrame(X_sel, index=X_raw.index)
X_sel = X_sel.astype(float)

evt = train[evt_col]
if not pd.api.types.is_numeric_dtype(evt):
    m = {}
    for v in evt.dropna().unique():
        sv = str(v).lower()
        m[v] = 1 if ('dead' in sv or 'deceas' in sv or 'died' in sv or sv.startswith('1')) else 0
    evt = evt.map(m)

X_fit = X_sel.copy()
X_fit[dur_col] = train[dur_col].astype(float).values
X_fit[evt_col] = evt.astype(float).values

cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
mlca_runtime.fit_record(cph, X_fit, "train", duration_col=dur_col, event_col=evt_col)
mlca_runtime.save_model(cph, "alternative_model.pkl")
print(f"rows={len(train)} feats={X_sel.shape[1]} cox folded")
