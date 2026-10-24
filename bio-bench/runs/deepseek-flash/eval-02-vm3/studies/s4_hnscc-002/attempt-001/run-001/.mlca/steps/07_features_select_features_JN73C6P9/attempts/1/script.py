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
group = protocol['group_column']
task = protocol['task']

event_col = targets[0]
for key in ['duration_column', 'time_column']:
    if key in protocol:
        duration_col = protocol[key]
        break
else:
    if task.get('time_columns'):
        duration_col = task['time_columns'][0]
    else:
        raise KeyError(f"Duration column missing. Available protocol keys: {list(protocol.keys())}")

train = pd.read_parquet(inputs['train'])
with open(inputs['06_preprocess_fit_preprocessing_GDNFB5M3/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = list(preprocessor.feature_names_in_)
else:
    exclude = set(targets) | set(excluded) | {group}
    feature_cols = []
    for c in train.columns:
        if c in exclude:
            continue
        if train[c].dtype == 'object' and train[c].nunique() == len(train):
            continue
        if train[c].dtype.kind in 'iu' and train[c].nunique() == len(train) and any(k in c.lower() for k in ['id','code','key','sample']):
            continue
        feature_cols.append(c)

event_raw = train[event_col]
if event_raw.dtype == 'object':
    event_series = event_raw.map({'Dead': 1, 'Alive': 0})
    if event_series.isna().any():
        raise ValueError(f"Unrecognized event values: {event_raw.unique()}")
else:
    event_series = event_raw

valid = train[duration_col].notna() & event_series.notna()
train = train[valid].copy()
event_series = event_series[valid]
train_duration = train[duration_col]

X_train = train[feature_cols]
X_train_pre = preprocessor.transform(X_train)
if isinstance(X_train_pre, pd.DataFrame):
    feature_names = list(X_train_pre.columns)
    is_sparse = False
else:
    is_sparse = True
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = list(preprocessor.get_feature_names_out())
    else:
        feature_names = [f'f{i}' for i in range(X_train_pre.shape[1])]

pvals = {}
for i, name in enumerate(feature_names):
    col_vals = X_train_pre[:, i].toarray().ravel() if is_sparse else X_train_pre[name].values
    df = pd.DataFrame({
        'duration': train_duration,
        'event': event_series,
        'feature': col_vals
    })
    cph = CoxPHFitter()
    try:
        cph.fit(df, duration_col='duration', event_col='event')
        pvals[name] = cph.summary.loc['feature', 'p']
    except Exception:
        pvals[name] = np.nan

selected = [c for c, p in pvals.items() if p < 0.10]
if len(selected) > 20:
    selected = sorted(selected, key=lambda c: pvals[c])[:20]

if is_sparse:
    selected_indices = [feature_names.index(c) for c in selected]
    selector_ct = ColumnTransformer(
        [('keep', 'passthrough', selected_indices)],
        remainder='drop'
    )
else:
    selector_ct = ColumnTransformer(
        [('keep', 'passthrough', selected)],
        remainder='drop'
    )
    selector_ct.set_output(transform='pandas')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', selector_ct)
])

mlca_runtime.fit_record(pipeline, X_train, 'train', y=None)
mlca_runtime.save_model(pipeline, 'selector.pkl')
mlca_runtime.record_metric('n_selected_features', float(len(selected)), 'train')
print(f"Selected {len(selected)} features out of {len(feature_names)} with p<0.10")
