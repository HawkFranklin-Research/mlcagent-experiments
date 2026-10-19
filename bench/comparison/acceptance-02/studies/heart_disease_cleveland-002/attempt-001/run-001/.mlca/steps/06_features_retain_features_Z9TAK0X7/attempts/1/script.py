import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column', None)
train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_5Z83A35F/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = list(preprocessor.feature_names_in_)
else:
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    feature_cols = [c for c in train.columns if c not in drop]
    for c in list(feature_cols):
        if train[c].dtype == 'object' or pd.api.types.is_string_dtype(train[c]):
            if train[c].nunique() == len(train):
                feature_cols.remove(c)

feature_cols = [c for c in feature_cols if c in train.columns and c not in targets and c not in excluded and c != group]
X_train = train[feature_cols].copy()

selector = FunctionTransformer()
mlca_runtime.fit_record(selector, X_train, partition='train')
mlca_runtime.save_model(selector, 'selector_all.pkl')
mlca_runtime.record_metric('n_features', len(feature_cols), 'train')
print(f"Fitted passthrough selector on {X_train.shape[0]} rows, {len(feature_cols)} features")
