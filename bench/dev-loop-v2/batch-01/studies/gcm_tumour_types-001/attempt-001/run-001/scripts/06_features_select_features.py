import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.frozen import FrozenEstimator

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
for key in ('targets', 'excluded_columns'):
    if key not in protocol:
        raise KeyError(f"missing '{key}'; available top-level keys: {list(protocol.keys())}")
targets = protocol['targets']
excluded = protocol['excluded_columns']

train = pd.read_parquet(inputs['train'])
with open(inputs['05_preprocess_fit_preprocessing_63H1R042/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

raw_cols = list(getattr(preprocessor, 'feature_names_in_', []))
if not raw_cols:
    raw_cols = [c for c in train.columns if c not in targets and c not in excluded]
missing = [c for c in raw_cols if c not in train.columns]
if missing:
    raise KeyError(f"missing feature columns: {missing}")

X_train = train[raw_cols]
y_train = train[targets[0]] if len(targets) == 1 else train[targets]

try:
    n_features = len(preprocessor.get_feature_names_out())
except Exception:
    n_features = int(preprocessor.transform(train[raw_cols].head(5)).shape[1])
k = max(1, min(20, n_features))

pipe = Pipeline([
    ('preprocessing', FrozenEstimator(preprocessor)),
    ('selector', SelectKBest(mutual_info_classif, k=k)),
])
if n_features <= 300:
    try:
        pipe.set_output(transform='pandas')
    except Exception:
        pass

mlca_runtime.fit_record(pipe, X_train, 'train', y=y_train)
mlca_runtime.save_model(pipe, 'selector.pkl')
mlca_runtime.record_metric('n_features_input', float(n_features), 'train')
mlca_runtime.record_metric('k_selected', float(k), 'train')
print(f'features={n_features} k={k} rows={len(X_train)}')
print('saved selector.pkl')
