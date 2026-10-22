import mlca_runtime, json, pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from lifelines import CoxPHFitter

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
if 'targets' not in protocol:
    raise KeyError(f"Missing targets. Available top-level keys: {list(protocol.keys())}")
targets = protocol['targets']
dur = targets[0]
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column', None)
task = protocol.get('task', {})

ev = None
for d in (protocol, task):
    if isinstance(d, dict):
        for k in ('event_column', 'event_col', 'event'):
            if k in d:
                ev = d[k]
                break
    if ev is not None:
        break
if ev is None:
    ev = "Overall Survival Status"

train = pd.read_parquet(inp['train'])
with open(inp['05_preprocess_fit_preprocessing_49B478YE/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inp['06_features_select_features_B4QNDNPN/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

def input_cols_of(obj):
    if obj is None:
        return None
    if hasattr(obj, 'feature_names_in_'):
        try:
            return list(obj.feature_names_in_)
        except Exception:
            return None
    return None

pre_in = input_cols_of(pre)
sel_in = input_cols_of(sel)
use_sel_only = False
if sel is not None and isinstance(sel, Pipeline):
    first = next(iter(sel.named_steps.values()))
    first_in = input_cols_of(first)
    if first_in is not None and pre_in is not None and list(first_in) == list(pre_in):
        use_sel_only = True
if not use_sel_only and sel_in is not None and pre_in is not None and list(sel_in) == list(pre_in):
    use_sel_only = True

exclude = set(targets) | set(excluded) | {ev}
if group:
    exclude.add(group)
n = len(train)
feat_cols = []
for c in train.columns:
    if c in exclude:
        continue
    s = train[c]
    if (pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)) and s.nunique(dropna=True) == n:
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == n and (s.is_monotonic_increasing or s.is_monotonic_decreasing):
        continue
    feat_cols.append(c)

if use_sel_only and sel_in is not None:
    input_cols = list(sel_in)
elif pre_in is not None:
    input_cols = list(pre_in)
else:
    input_cols = feat_cols

bad = set(targets) | set(excluded) | {ev}
if group:
    bad.add(group)
if any(c in bad for c in input_cols):
    raise ValueError("Saved frontend expects an outcome/excluded column")

def get_names(obj, cols, width):
    for args in ((), (cols,)):
        try:
            names = obj.get_feature_names_out(*args)
            if names is not None and len(names) == width:
                return list(names)
        except Exception:
            pass
    return [f"f{i}" for i in range(width)]

def transform(obj, df, cols):
    out = obj.transform(df[cols])
    if isinstance(out, pd.DataFrame):
        return out
    if hasattr(out, 'toarray'):
        out = out.toarray()
    out = np.asarray(out)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    return pd.DataFrame(out, columns=get_names(obj, cols, out.shape[1]), index=df.index)

if use_sel_only:
    X_feat = transform(sel, train, input_cols)
    pre_saved, sel_saved = sel, None
else:
    X_pre = transform(pre, train, input_cols)
    if sel is not None:
        out = sel.transform(X_pre)
        if isinstance(out, pd.DataFrame):
            X_feat = out
        else:
            if hasattr(out, 'toarray'):
                out = out.toarray()
            out = np.asarray(out)
            if out.ndim == 1:
                out = out.reshape(-1, 1)
            X_feat = pd.DataFrame(out, columns=get_names(sel, list(X_pre.columns), out.shape[1]), index=train.index)
    else:
        X_feat = X_pre
    pre_saved, sel_saved = pre, sel

def encode_event(s):
    vals = pd.unique(s.dropna())
    if pd.api.types.is_numeric_dtype(s) and set(vals).issubset({0, 1, 0.0, 1.0}):
        return s.astype(int)
    mp = {}
    for v in vals:
        lv = str(v).strip().lower()
        if lv in {'dead', 'deceased', 'event', '1', 'true', 'yes', 'progressed', 'recurred'}:
            mp[v] = 1
        elif lv in {'alive', 'censored', '0', 'false', 'no', 'living'}:
            mp[v] = 0
        else:
            raise ValueError(f"Unmapped event value: {v!r}")
    return s.map(mp)

event_s = encode_event(train[ev])
mask = train[dur].notna() & event_s.notna()
X_fit = X_feat.loc[mask].copy()
X_fit[dur] = train.loc[mask, dur]
X_fit[ev] = event_s.loc[mask].astype(int)

cph = CoxPHFitter(penalizer=0.0)
cph = mlca_runtime.fit_record(cph, X_fit, 'train', duration_col=dur, event_col=ev)
model = {
    'preprocess': pre_saved,
    'selector': sel_saved,
    'coxph': cph,
    'input_columns': input_cols,
    'feature_names': list(X_feat.columns),
    'duration_col': dur,
    'event_col': ev,
}
mlca_runtime.save_model(model, 'model.pkl')

ci = getattr(cph, 'concordance_index_', None)
if ci is not None:
    mlca_runtime.record_metric('train_c_index', float(ci), 'train')
print(f"train rows used: {len(X_fit)}")
print(f"transformed features: {X_feat.shape[1]}")
if ci is not None:
    print(f"train c-index: {ci:.4f}")
