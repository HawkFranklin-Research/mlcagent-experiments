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
task = protocol['task']
event_col = targets[0]

duration_col = None
for key in ['duration_column', 'time_column']:
    if key in protocol:
        duration_col = protocol[key]
        break
if duration_col is None and task.get('time_columns'):
    duration_col = task['time_columns'][0]
if duration_col is None:
    raise KeyError(f"Duration column missing. Available protocol keys: {list(protocol.keys())}")

train = pd.read_parquet(inputs['train'])
with open(inputs['06_preprocess_fit_preprocessing_GDNFB5M3/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

exclude = set(targets) | set(excluded) | {duration_col}
if group is not None:
    exclude.add(group)
if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = list(preprocessor.feature_names_in_)
else:
    feature_cols = []
    for c in train.columns:
        if c in exclude:
            continue
        s = train[c]
        if s.dtype == 'object' and s.nunique() == len(train):
            continue
        if s.dtype.kind in 'iu' and s.nunique() == len(train) and any(k in c.lower() for k in ['id', 'code', 'key', 'sample']):
            continue
        feature_cols.append(c)

duration_series = pd.to_numeric(train[duration_col], errors='coerce')
event_num = pd.to_numeric(train[event_col], errors='coerce')
if event_num.notna().sum() == 0:
    ev = train[event_col].astype(str).str.lower()
    event_series = pd.Series(
        np.where(ev.str.contains('dead|deceased'), 1.0, np.where(ev.str.contains('alive|living'), 0.0, np.nan)),
        index=train.index,
    )
else:
    event_series = event_num

valid = duration_series.notna() & event_series.notna()
train = train[valid].copy()
duration_series = duration_series[valid]
event_series = event_series[valid]
X_train = train[feature_cols]

X_train_pre = preprocessor.transform(X_train)
if isinstance(X_train_pre, pd.DataFrame):
    output_is_frame = True
    feature_names = list(X_train_pre.columns)
else:
    output_is_frame = False
    feature_names = list(preprocessor.get_feature_names_out()) if hasattr(preprocessor, 'get_feature_names_out') else [f'f{i}' for i in range(X_train_pre.shape[1])]

pvals = {}
for i, name in enumerate(feature_names):
    if output_is_frame:
        col_vals = X_train_pre.iloc[:, i].to_numpy()
    else:
        col = X_train_pre[:, i]
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
        pvals[name] = fitted.summary.loc['feature', 'p']
    except Exception:
        pvals[name] = np.nan

selected = [c for c, p in pvals.items() if p < 0.10]
if len(selected) > 20:
    selected = sorted(selected, key=lambda c: pvals[c])[:20]

if output_is_frame:
    selector_ct = ColumnTransformer([('keep', 'passthrough', selected)], remainder='drop')
    selector_ct.set_output(transform='pandas')
else:
    selected_indices = [feature_names.index(c) for c in selected]
    selector_ct = ColumnTransformer([('keep', 'passthrough', selected_indices)], remainder='drop')

pipeline = Pipeline([('preprocessor', preprocessor), ('selector', selector_ct)])
mlca_runtime.fit_record(pipeline, X_train, 'train', y=None)
mlca_runtime.save_model(pipeline, 'selector.pkl')
mlca_runtime.record_metric('n_selected_features', float(len(selected)), 'train')
print(f"Selected {len(selected)} of {len(feature_names)} transformed features (p<0.10).")
