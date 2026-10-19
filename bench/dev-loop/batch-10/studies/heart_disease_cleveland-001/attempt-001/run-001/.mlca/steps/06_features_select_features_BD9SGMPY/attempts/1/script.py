import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = list(protocol['targets'])
excluded = list(protocol.get('excluded_columns', []))
group_col = protocol.get('group_column', None)

train = pd.read_parquet(inputs['train'])
drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)

with open(inputs['05_preprocess_fit_preprocessing_8FQ40592/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, 'feature_names_in_'):
    features = list(preprocessor.feature_names_in_)
else:
    features = []
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        nun = s.dropna()
        if len(nun) == 0:
            continue
        if nun.is_unique and nun.dtype.kind in 'iuO':
            continue
        features.append(c)

leak = [c for c in features if c in drop]
if leak:
    raise RuntimeError(f'feature list contains targets/excluded columns: {leak}')
missing = [c for c in features if c not in train.columns]
if missing:
    raise RuntimeError(f'feature columns missing from train: {missing}')

X_train = train[features]
y_train = train[targets[0]]

Xt = preprocessor.transform(X_train)
if hasattr(Xt, 'toarray'):
    Xt = Xt.toarray()
n_feat = Xt.shape[1]
k = max(1, min(20, n_feat))

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', SelectKBest(mutual_info_classif, k=k)),
])

mlca_runtime.fit_record(pipe, X_train, 'train', y_train)
mlca_runtime.save_model(pipe, 'selector.pkl')

mlca_runtime.record_metric('n_features_input', float(len(features)), 'train')
mlca_runtime.record_metric('n_features_preprocessed', float(n_feat), 'train')
mlca_runtime.record_metric('k_selected', float(k), 'train')
print(f'input={len(features)} preprocessed={n_feat} selected={k} rows={len(train)}')
