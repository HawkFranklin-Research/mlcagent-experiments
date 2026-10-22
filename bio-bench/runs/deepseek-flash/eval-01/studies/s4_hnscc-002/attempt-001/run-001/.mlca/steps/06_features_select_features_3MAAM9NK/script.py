import json, pickle
import numpy as np, pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from lifelines import CoxPHFitter

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
for k in ['targets', 'excluded_columns', 'group_column', 'task']:
    if k not in protocol:
        raise KeyError(f"Missing protocol key {k}; available top-level keys: {list(protocol.keys())}")
targets = protocol['targets']
excluded = set(protocol['excluded_columns'])
group = protocol['group_column']

train = pd.read_parquet(inp['train'])
with open(inp['05_preprocess_fit_preprocessing_QC6SFFG9/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
if not hasattr(pre, 'transform'):
    raise TypeError('preprocessor.pkl has no transform')

if hasattr(pre, 'feature_names_in_'):
    feat_cols = list(pre.feature_names_in_)
    bad = set(feat_cols) & (set(targets) | excluded | ({group} if group is not None else set()))
    if bad:
        raise ValueError(f'Forbidden columns in preprocessor input: {sorted(bad)}')
else:
    def key_like(c):
        s = train[c]
        if pd.api.types.is_float_dtype(s):
            return False
        if pd.api.types.is_integer_dtype(s):
            return len(s) > 2 and s.nunique(dropna=True) == len(s)
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            return s.nunique(dropna=True) == len(s)
        return False
    feat_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group and not key_like(c)]

X_raw = train[feat_cols].copy()
Xt = pre.transform(X_raw)
is_df = isinstance(Xt, pd.DataFrame)
if is_df:
    names = list(Xt.columns)
else:
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = [f'x{i}' for i in range(Xt.shape[1])]
if len(names) != Xt.shape[1]:
    names = [f'x{i}' for i in range(Xt.shape[1])]

binary = [c for c in targets if train[c].nunique(dropna=True) <= 2]
numeric = [c for c in targets if pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique(dropna=True) > 2]
if len(binary) != 1 or not numeric:
    raise ValueError(f'Cannot identify event/duration from protocol targets; binary={binary}, numeric={numeric}')
event_col = binary[0]
duration_col = numeric[0]
dur = pd.to_numeric(train[duration_col], errors='coerce')
ev = train[event_col]
if pd.api.types.is_numeric_dtype(ev):
    ev = (ev.fillna(0) > 0).astype(int)
else:
    ev = ev.astype(str).str.lower().str.contains('dead|deceased|event|positive|progress|1|true|yes', regex=True).astype(int)
valid = (dur.notna() & ev.notna()).to_numpy()
dur_v = dur.to_numpy()[valid]
ev_v = ev.to_numpy()[valid]
idx_v = train.index[valid]

ps = []
for i in range(Xt.shape[1]):
    try:
        if is_df:
            vals = Xt.iloc[:, i].to_numpy()
        else:
            col = Xt[:, i]
            vals = col.toarray().ravel() if hasattr(col, 'toarray') else np.asarray(col).ravel()
        sd = np.nanstd(vals.astype(float))
        if not np.isfinite(sd) or sd == 0:
            continue
        df = pd.DataFrame({'x': vals[valid], duration_col: dur_v, event_col: ev_v}, index=idx_v)
        cph = mlca_runtime.fit_record(CoxPHFitter(), df, 'train', duration_col=duration_col, event_col=event_col)
        p = float(cph.summary.loc['x', 'p'])
        if np.isfinite(p):
            ps.append((p, i))
    except Exception:
        continue

ps.sort()
selected = [i for p, i in ps if p < 0.10][:20]
if is_df:
    sel_cols = [(f'c{i}', 'passthrough', [names[i]]) for i in selected]
else:
    sel_cols = [(f'c{i}', 'passthrough', [int(i)]) for i in selected]
if sel_cols:
    selector = ColumnTransformer(sel_cols, remainder='drop', verbose_feature_names_out=False)
else:
    drop_cols = [('drop_all', 'drop', names if is_df else list(range(Xt.shape[1])))]
    selector = ColumnTransformer(drop_cols, remainder='drop', verbose_feature_names_out=False)
selector.set_output(transform='pandas')
final = Pipeline([('pre', pre), ('select', selector)])
final.set_output(transform='pandas')
mlca_runtime.fit_record(final, X_raw, 'train')
mlca_runtime.save_model(final, 'selector.pkl')
mlca_runtime.record_metric('n_selected_covariates', float(len(selected)), 'train')
if ps:
    mlca_runtime.record_metric('min_univariable_cox_p', float(ps[0][0]), 'train')
print(f"train_rows={len(train)} features={len(feat_cols)} transformed={Xt.shape[1]}")
print(f"event={event_col} duration={duration_col} selected={len(selected)}")
