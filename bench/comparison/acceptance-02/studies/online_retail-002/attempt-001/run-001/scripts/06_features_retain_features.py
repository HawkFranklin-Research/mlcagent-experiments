import json, pickle
import mlca_runtime
import pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
train = pd.read_parquet(inp['train'])

targets = list(protocol['targets'])
excluded = list(protocol['excluded_columns'])
group = protocol['group_column'] if 'group_column' in protocol else None
if 'group_column' not in protocol:
    print('available protocol keys:', list(protocol.keys()))

with open(inp['05_preprocess_fit_preprocessing_MATQH9FA/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

feature_cols = list(getattr(pre, 'feature_names_in_', []))
if not feature_cols:
    drop = set(targets) | set(excluded) | ({group} if group else set())
    feature_cols = [c for c in train.columns if c not in drop]
    key_like = []
    for c in feature_cols:
        s = train[c]
        if pd.api.types.is_integer_dtype(s) or s.dtype == object:
            nonnull = s.notna().sum()
            if nonnull and s.nunique(dropna=True) == nonnull:
                key_like.append(c)
    feature_cols = [c for c in feature_cols if c not in key_like]

forbidden = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in feature_cols if c not in forbidden]
X = train[feature_cols].copy()

steps = list(pre.steps) if isinstance(pre, Pipeline) else [('preprocessor', pre)]
selector = Pipeline(steps + [('passthrough', 'passthrough')])

mlca_runtime.fit_record(selector, X, 'train')
mlca_runtime.save_model(selector, 'selector_all.pkl')
print('rows', X.shape[0], 'features', X.shape[1])
