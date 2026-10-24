import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets, excluded = protocol['targets'], protocol['excluded_columns']
group = protocol.get('group_column')
duration_col, event_col = targets[0], targets[1]
train = pd.read_parquet(inputs['train'])

def is_keylike(s):
    n = s.dropna().nunique()
    if n == 0:
        return False
    if s.dtype == object:
        return n == len(s.dropna()) and len(s.dropna()) > 20
    if pd.api.types.is_integer_dtype(s):
        return n == len(s.dropna()) and s.max() - s.min() + 1 == n
    return False

drop = set(targets) | set(excluded) | ({group} if group else set())
feat_cols = [c for c in train.columns if c not in drop and not is_keylike(train[c])]

with open(inputs['05_preprocess_fit_preprocessing_RJBK5NR5/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

pre_cols = list(pre.feature_names_in_) if hasattr(pre, 'feature_names_in_') else feat_cols
X_train = train[pre_cols].copy()

def to_event(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float).values
    m = {'dead':1,'deceased':1,'alive':0,'1':1,'0':0,'true':1,'false':0,'yes':1,'no':0}
    return s.astype(str).str.lower().str.strip().map(m).values

event = to_event(train[event_col])
time_v = pd.to_numeric(train[duration_col], errors='coerce').values

try:
    X_pre = pre.transform(X_train)
except Exception:
    pre.fit(X_train)
    X_pre = pre.transform(X_train)
if hasattr(X_pre, 'toarray'):
    X_pre = X_pre.toarray()
if not isinstance(X_pre, pd.DataFrame):
    arr = np.asarray(X_pre)
    X_pre = pd.DataFrame(arr, columns=[f'f{i}' for i in range(arr.shape[1])])
feat_names = list(X_pre.columns)

valid = ~pd.isna(time_v) & ~pd.isna(event)
base = pd.DataFrame({'T': time_v, 'E': event}).loc[valid].reset_index(drop=True)
Xp = X_pre.loc[valid].reset_index(drop=True)

pvals = {}
for c in feat_names:
    xcol = pd.to_numeric(Xp[c], errors='coerce')
    mask = xcol.notna().values
    if mask.sum() < 20 or xcol[mask].nunique() < 2:
        pvals[c] = 1.0
        continue
    df = base.loc[mask].copy()
    df['x'] = xcol[mask].values
    try:
        cph = CoxPHFitter()
        cph.fit(df, duration_col='T', event_col='E')
        pvals[c] = float(cph.summary.loc['x', 'p'])
    except Exception:
        pvals[c] = 1.0

ranked = sorted(pvals.items(), key=lambda kv: kv[1])
selected = [c for c, p in ranked if p < 0.10][:20] or [c for c, _ in ranked[:20]]
if isinstance(X_pre, pd.DataFrame):
    sel_spec = selected
else:
    idx_map = {c: i for i, c in enumerate(feat_names)}
    sel_spec = [idx_map[c] for c in selected]

selector_ct = ColumnTransformer([('keep', 'passthrough', sel_spec)], remainder='drop')
final = Pipeline([('pre', pre), ('select', selector_ct)])
try:
    final.set_output(transform='pandas')
except Exception:
    pass

mlca_runtime.fit_record(final, X_train, 'train', y=None)
mlca_runtime.save_model(final, 'selector.pkl')
mlca_runtime.record_metric('n_selected', float(len(selected)), 'train')
print(f'rows_train={len(X_train)} feat_in={len(feat_names)} selected={len(selected)}')
