import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
ins = mlca_runtime.inputs()
proto = json.loads(ins['_protocol'].read_text())
targets = proto['targets']
excluded = set(proto.get('excluded_columns', []))
group = proto.get('group_column')
train = pd.read_parquet(ins['train'])

reserved = set(targets) | excluded | ({group} if group else set())
feat_cols = []
for c in train.columns:
    if c in reserved:
        continue
    s = train[c]
    if s.isna().all():
        continue
    if (s.dtype == object or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
        continue
    feat_cols.append(c)

with open(ins['05_preprocess_fit_preprocessing_5QSFC5J4/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

if hasattr(pre, 'feature_names_in_'):
    want = list(pre.feature_names_in_)
    leak = reserved & set(want)
    if leak:
        raise ValueError(f'preprocessor input leaks reserved columns: {sorted(leak)}')
    missing = [c for c in want if c not in feat_cols]
    if missing:
        raise ValueError(f'preprocessor inputs not in feature columns: {missing}')
    feat_cols = want

n_features = len(feat_cols)
k = min(20, n_features)
selector = SelectKBest(score_func=mutual_info_classif, k=k)

if isinstance(pre, Pipeline):
    pipe = Pipeline(pre.steps + [('select', selector)])
else:
    pipe = Pipeline([('pre', pre), ('select', selector)])
try:
    pipe.set_output(transform='pandas')
except Exception:
    pass

X = train[feat_cols]
y = train[targets[0]]
fitted = mlca_runtime.fit_record(pipe, X, 'train', y=y)
mlca_runtime.save_model(fitted, 'selector.pkl')
print('train_rows', len(train), 'n_features', n_features, 'k', k)
print('saved selector.pkl')
