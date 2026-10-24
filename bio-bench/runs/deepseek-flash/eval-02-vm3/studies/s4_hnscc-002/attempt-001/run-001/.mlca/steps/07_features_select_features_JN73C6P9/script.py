import mlca_runtime
import pandas as pd
import numpy as np
import pickle
import json
from lifelines import CoxPHFitter
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')
task = protocol.get('task', {}) or {}
event_col = targets[0]

train = pd.read_parquet(inputs['train'])
with open(inputs['06_preprocess_fit_preprocessing_GDNFB5M3/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

# Resolve duration column
duration_col = None
for src in [protocol, task, protocol.get('columns', {}) or {}, task.get('columns', {}) or {}]:
    if not isinstance(src, dict):
        continue
    for k in ['duration_column', 'time_column', 'duration', 'time']:
        v = src.get(k)
        if isinstance(v, str) and v in train.columns:
            duration_col = v
            break
    if duration_col:
        break
if duration_col is None and task.get('time_columns'):
    cand = task['time_columns'][0]
    if cand in train.columns:
        duration_col = cand
if duration_col is None:
    excl_set = set(targets) | set(excluded)
    for pat in ['overall survival (months)', 'overall survival months', 'os_months',
                'os.months', 'survival_months', 'survival month', 'months', 'time-to']:
        cands = [c for c in train.columns if c not in excl_set and pat in c.lower()]
        if len(cands) == 1:
            duration_col = cands[0]
            break
if duration_col is None:
    raise KeyError(f"Duration column not found. Available protocol keys: {list(protocol.keys())}")

exclude = set(targets) | set(excluded) | {duration_col}
if group is not None:
    exclude.add(group)

if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = list(preprocessor.feature_names_in_)
else:
    n = len(train)
    feature_cols = []
    for c in train.columns:
        if c in exclude:
            continue
        s = train[c]
        if s.dtype == object and s.nunique() == n:
            continue
        if s.dtype.kind in 'iu' and s.nunique() == n and any(k in c.lower() for k in ['id', 'code', 'key', 'sample']):
            continue
        feature_cols.append(c)

duration_series = pd.to_numeric(train[duration_col], errors='coerce')
event_num = pd.to_numeric(train[event_col], errors='coerce')
if event_num.notna().sum() == 0:
    ev = train[event_col].astype(str).str.lower()
    event_series = pd.Series(
        np.where(ev.str.contains('dead|deceased'), 1.0,
                 np.where(ev.str.contains('alive|living'), 0.0, np.nan)),
        index=train.index,
    )
else:
    event_series = event_num

valid = duration_series.notna() & event_series.notna()
train = train[valid].copy()
duration_series = duration_series[valid]
event_series = event_series[valid]
X_train = train[feature_cols]

X_pre = preprocessor.transform(X_train)
if isinstance(X_pre, pd.DataFrame):
    output_is_frame = True
    feature_names = list(X_pre.columns)
else:
    output_is_frame = False
    if hasattr(preprocessor, 'get_feature_names_out'):
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = [f'f{i}' for i in range(X_pre.shape[1])]
    else:
        feature_names = [f'f{i}' for i in range(X_pre.shape[1])]

pvals = {}
for i, name in enumerate(feature_names):
    if output_is_frame:
        col_vals = X_pre.iloc[:, i].to_numpy()
    else:
        col = X_pre[:, i]
        col_vals = col.toarray().ravel() if hasattr(col, 'toarray') else np.asarray(col).ravel()
    df = pd.DataFrame({
        duration_col: duration_series.to_numpy(),
        event_col: event_series.to_numpy(),
        'feature': col_vals,
    })
    try:
        fitted = mlca_runtime.fit_record(
            CoxPHFitter(), df, 'train',
            duration_col=duration_col, event_col=event_col,
        )
        pvals[name] = float(fitted.summary.loc['feature', 'p'])
    except Exception:
        pvals[name] = np.nan

selected = [c for c, p in pvals.items() if p is not None and np.isfinite(p) and p < 0.10]
if len(selected) > 20:
    selected = sorted(selected, key=lambda c: pvals[c])[:20]

if output_is_frame:
    selector_ct = ColumnTransformer([('keep', 'passthrough', selected)], remainder='drop')
    try:
        selector_ct.set_output(transform='pandas')
    except Exception:
        pass
else:
    sel_idx = [feature_names.index(c) for c in selected]
    selector_ct = ColumnTransformer([('keep', 'passthrough', sel_idx)], remainder='drop')

pipeline = Pipeline([('preprocessor', preprocessor), ('selector', selector_ct)])
mlca_runtime.fit_record(pipeline, X_train, 'train', y=None)
mlca_runtime.save_model(pipeline, 'selector.pkl')
mlca_runtime.record_metric('n_selected_features', float(len(selected)), 'train')
print(f"Selected {len(selected)} of {len(feature_names)} transformed features (p<0.10).")
