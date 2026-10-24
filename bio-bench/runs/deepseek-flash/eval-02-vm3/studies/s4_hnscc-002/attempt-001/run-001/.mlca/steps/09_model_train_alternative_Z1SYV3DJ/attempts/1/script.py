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
    raise ValueError(f"Expected survival task, got {task}")

train = pd.read_parquet(inputs['train'])
duration_col = 'Overall Survival (Months)'
event_col = protocol['targets'][0]

with open(inputs['06_preprocess_fit_preprocessing_GDNFB5M3/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['07_features_select_features_JN73C6P9/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

# If selector already contains preprocessing, use it alone
if isinstance(sel, Pipeline) and len(sel.named_steps) > 1:
    transform = sel
else:
    transform = Pipeline([('pre', pre), ('sel', sel)])

raw_features = list(getattr(pre, 'feature_names_in_', None) or [c for c in train.columns if c not in targets and c not in excluded and c != group])
raw_features = [c for c in raw_features if c not in targets and c not in excluded]

X = transform.transform(train[raw_features])
if not isinstance(X, pd.DataFrame):
    names = transform.get_feature_names_out()
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = pd.DataFrame(X, index=train.index, columns=names)

y = train[event_col]
if y.dtype == object:
    y = y.map({'Dead': 1, 'Alive': 0, 'dead': 1, 'alive': 0})
y = y.astype(int)

df = X.copy()
df[duration_col] = train[duration_col].astype(float)
df[event_col] = y.values

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
print(f"Saved alternative_model.pkl with {len(raw_features)} raw features -> {X.shape[1]} selected features")
