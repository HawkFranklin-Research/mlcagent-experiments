import mlca_runtime, json, pickle
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')
task = protocol['task']
if task.get('type') != 'survival':
    raise ValueError(f"Expected survival task, got {task.get('type')}")

train = pd.read_parquet(inputs['train'])
event_col = targets[0]
duration_col = (protocol.get('duration_column')
                or task.get('duration_column')
                or (task.get('time_columns') or [None])[0]
                or 'Overall Survival (Months)')

with open(inputs['06_preprocess_fit_preprocessing_GDNFB5M3/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['07_features_select_features_JN73C6P9/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

if isinstance(sel, Pipeline) and len(sel.named_steps) > 1:
    transform = sel
else:
    transform = Pipeline([('pre', pre), ('sel', sel)])

fn = getattr(pre, 'feature_names_in_', None)
if fn is None:
    fn = getattr(sel, 'feature_names_in_', None)
if fn is not None:
    raw_features = [str(c) for c in list(fn)]
else:
    raw_features = [str(c) for c in train.columns]
raw_features = [c for c in raw_features
                if c not in targets and c not in excluded
                and c != group and c != duration_col]

def _flat(names):
    return ['__'.join(str(x) for x in n) if isinstance(n, tuple) else str(n) for n in names]

X = transform.transform(train[raw_features])
if isinstance(X, pd.DataFrame):
    X = X.copy()
    X.columns = _flat(X.columns)
    X.index = train.index
else:
    names = (_flat(transform.get_feature_names_out())
             if hasattr(transform, 'get_feature_names_out')
             else [f'f{i}' for i in range(X.shape[1])])
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = pd.DataFrame(X, index=train.index, columns=names)

ev_raw = train[event_col]
if ev_raw.dtype == object or str(ev_raw.dtype) == 'string':
    ev = ev_raw.map({'Dead': 1, 'Alive': 0, 'dead': 1, 'alive': 0,
                     'DECEASED': 1, 'LIVING': 0})
    if ev.isna().any():
        ev = pd.to_numeric(ev_raw, errors='coerce')
else:
    ev = pd.to_numeric(ev_raw, errors='coerce')
if ev.isna().any():
    raise ValueError(f"Unmapped event values in {event_col}")
ev = ev.astype(int)

dur = pd.to_numeric(train[duration_col], errors='coerce')
if dur.isna().any():
    raise ValueError(f"Non-numeric values in {duration_col}")

df = X.copy()
df.columns = _flat(df.columns)
drop_cols = [c for c in (duration_col, event_col) if c in df.columns]
if drop_cols:
    df = df.drop(columns=drop_cols)
if df.columns.duplicated().any():
    df = df.loc[:, ~df.columns.duplicated()]
df[duration_col] = dur.to_numpy()
df[event_col] = ev.to_numpy()
if duration_col not in df.columns or event_col not in df.columns:
    raise RuntimeError(f"duration/event columns missing: dur={duration_col in df.columns} evt={event_col in df.columns}")

cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
mlca_runtime.fit_record(cph, df, 'train', duration_col=duration_col, event_col=event_col)

bundle = {
    'preprocessor': pre,
    'selector': sel,
    'model': cph,
    'raw_features': raw_features,
    'duration_col': duration_col,
    'event_col': event_col,
}
mlca_runtime.save_model(bundle, 'alternative_model.pkl')
print(f"Saved alternative_model.pkl: {len(raw_features)} raw features -> {X.shape[1]} encoded")
