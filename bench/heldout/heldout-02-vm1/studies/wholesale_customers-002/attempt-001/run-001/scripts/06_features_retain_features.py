import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

train = pd.read_parquet(inputs['train'])
with open(inputs['05_preprocess_fit_preprocessing_WG30AQ6C/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

targets = list(protocol['targets'])
excluded = list(protocol['excluded_columns'])
group = protocol.get('group_column')

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
kept = []
for c in feature_cols:
    s = train[c]
    is_id = False
    if s.dtype.kind in ('O', 'U', 'S'):
        is_id = s.nunique(dropna=False) == len(s)
    elif s.dtype.kind in ('i', 'u'):
        if s.nunique(dropna=False) == len(s) and (s.max() - s.min() + 1) == len(s):
            is_id = True
    if not is_id:
        kept.append(c)

pre_cols = list(pre.feature_names_in_) if hasattr(pre, 'feature_names_in_') else kept
X = train[pre_cols]

selector_all = Pipeline([('preprocess', pre), ('select', 'passthrough')])
mlca_runtime.fit_record(selector_all, X, 'train')
mlca_runtime.save_model(selector_all, 'selector_all.pkl')
try:
    n_out = len(selector_all.get_feature_names_out())
except Exception:
    n_out = -1
print(f"feature_cols={len(kept)} fitted_in={len(pre_cols)} out={n_out}")
