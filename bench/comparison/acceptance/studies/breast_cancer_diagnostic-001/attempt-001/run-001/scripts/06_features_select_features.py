import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

for key in ('targets', 'excluded_columns', 'group_column', 'task'):
    if key not in protocol:
        raise KeyError(f"Missing protocol key '{key}'. Available: {sorted(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol['group_column']
target = targets[0] if len(targets) == 1 else targets

train = pd.read_parquet(inputs['train'])

skip = set(targets) | set(excluded)
if group:
    skip.add(group)
feat_cols = []
for c in train.columns:
    if c in skip:
        continue
    s = train[c]
    if s.dtype == object or str(s.dtype) == 'string':
        if s.nunique() == len(s) and s.notna().all():
            continue
    if pd.api.types.is_integer_dtype(s):
        if s.nunique() == len(s) and s.notna().all():
            if s.max() - s.min() == len(s) - 1:
                continue
    feat_cols.append(c)

with open(inputs['05_preprocess_fit_preprocessing_WZN8YG48/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, 'feature_names_in_'):
    feat_cols = list(preprocessor.feature_names_in_)

X = train[feat_cols].copy()
y = train[target]

try:
    n_out = preprocessor.transform(X.head(5)).shape[1]
except Exception:
    n_out = X.shape[1]
k = min(20, n_out)

pipe = Pipeline([
    ('pre', preprocessor),
    ('select', SelectKBest(mutual_info_classif, k=k)),
])

mlca_runtime.fit_record(pipe, X, 'train', y=y)
mlca_runtime.save_model(pipe, 'selector.pkl')
mlca_runtime.record_metric('n_features_in', X.shape[1], 'train')
mlca_runtime.record_metric('k_selected', k, 'train')
print(f"features in={X.shape[1]} selected={k}")
