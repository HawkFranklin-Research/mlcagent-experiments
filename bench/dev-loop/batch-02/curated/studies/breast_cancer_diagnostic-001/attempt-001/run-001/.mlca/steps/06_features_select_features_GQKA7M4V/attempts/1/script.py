import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

for key in ['targets', 'excluded_columns', 'group_column', 'task']:
    if key not in protocol:
        raise ValueError(f"Missing key {key} in _protocol. Available: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns'] or []
group = protocol['group_column']
if len(targets) != 1:
    raise ValueError(f"Expected exactly one target, got {targets}")
target = targets[0]

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_BKKYG4P6/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

if hasattr(pre, 'feature_names_in_'):
    feature_cols = list(pre.feature_names_in_)
    bad = set(feature_cols) & (set(targets) | set(excluded) | ({group} if group else set()))
    if bad:
        raise ValueError(f"Preprocessor expects excluded/target columns: {bad}")
else:
    drop = set(targets) | set(excluded) | ({group} if group else set())
    feature_cols = []
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        if not pd.api.types.is_numeric_dtype(s):
            if s.nunique(dropna=False) == len(s) and len(s) > 1:
                continue
        elif pd.api.types.is_integer_dtype(s):
            if s.nunique(dropna=False) == len(s) and len(s) > 1 and (s.max() - s.min() + 1) == len(s):
                continue
        feature_cols.append(c)

X = train[feature_cols].copy()
y = train[target]
y_enc = y if pd.api.types.is_numeric_dtype(y) else LabelEncoder().fit_transform(y)

if hasattr(pre, 'set_output'):
    try:
        pre.set_output(transform='pandas')
    except Exception:
        pass

n_features = len(feature_cols)
if hasattr(pre, 'n_features_in_'):
    try:
        n_features = pre.transform(X).shape[1]
    except Exception:
        pass
k = min(20, n_features)

pipeline = Pipeline([
    ('preprocessor', pre),
    ('selector', SelectKBest(score_func=mutual_info_classif, k=k))
])

mlca_runtime.fit_record(pipeline, X, 'train', y=y_enc)
selected = int(pipeline.named_steps['selector'].get_support().sum())
mlca_runtime.record_metric('n_features_selected', float(selected), 'train')
mlca_runtime.save_model(pipeline, 'selector.pkl')

print(f"train rows: {len(train)}")
print(f"features: {len(feature_cols)}")
print(f"k: {k}, selected: {selected}")
