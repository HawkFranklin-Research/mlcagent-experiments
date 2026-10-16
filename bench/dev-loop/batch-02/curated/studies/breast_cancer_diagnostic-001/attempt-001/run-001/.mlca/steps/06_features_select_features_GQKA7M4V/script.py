import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

required = ['targets', 'excluded_columns', 'group_column', 'task']
missing = [k for k in required if k not in protocol]
if missing:
    raise ValueError(f"Missing keys {missing} in _protocol. Available: {list(protocol.keys())}")

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

bad = set(feature_cols) & (set(targets) | set(excluded) | ({group} if group else set()))
if bad:
    raise ValueError(f"Feature columns include target/excluded/group: {bad}")

X = train[feature_cols].copy()
y_enc = pd.factorize(train[target])[0]

if hasattr(pre, 'set_output'):
    try:
        pre.set_output(transform='pandas')
    except Exception:
        pass

n_features = len(feature_cols)
if hasattr(pre, 'transform'):
    try:
        n_features = pre.transform(X).shape[1]
    except Exception:
        pass

k = min(20, n_features)

if isinstance(pre, Pipeline):
    steps = list(pre.steps)
    selector_name = 'selector'
    if any(name == selector_name for name, _ in steps):
        selector_name = 'selectkbest'
    steps.append((selector_name, SelectKBest(score_func=mutual_info_classif, k=k)))
    pipeline = Pipeline(steps)
else:
    selector_name = 'selector'
    pipeline = Pipeline([
        ('preprocessor', pre),
        (selector_name, SelectKBest(score_func=mutual_info_classif, k=k))
    ])

mlca_runtime.fit_record(pipeline, X, 'train', y=y_enc)
selected = int(pipeline.named_steps[selector_name].get_support().sum())
mlca_runtime.record_metric('n_features_selected', float(selected), 'train')
mlca_runtime.save_model(pipeline, 'selector.pkl')

print(f"train rows: {len(train)}")
print(f"features: {len(feature_cols)}")
print(f"k: {k}, selected: {selected}")
