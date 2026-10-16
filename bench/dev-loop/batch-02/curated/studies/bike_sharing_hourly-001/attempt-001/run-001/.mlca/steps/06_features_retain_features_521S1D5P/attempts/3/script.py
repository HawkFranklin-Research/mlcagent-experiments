import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])
with open(inputs['05_preprocess_fit_preprocessing_KDGTX9T9/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = list(preprocessor.feature_names_in_)
else:
    targets = protocol.get('targets', [])
    excluded = protocol.get('excluded_columns', [])
    group = protocol.get('group_column')
    feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]

X_train = train[feature_cols].copy()
y_train = train[protocol['targets'][0]] if protocol.get('targets') else None

X_train_trans = preprocessor.transform(X_train)
if not isinstance(X_train_trans, pd.DataFrame):
    if hasattr(preprocessor, 'get_feature_names_out'):
        cols = preprocessor.get_feature_names_out()
    else:
        cols = [f'f{i}' for i in range(X_train_trans.shape[1])]
    X_train_trans = pd.DataFrame(X_train_trans, columns=cols, index=X_train.index)

selector = FunctionTransformer()
mlca_runtime.fit_record(selector, X_train_trans, 'train', y_train)
mlca_runtime.save_model(selector, 'selector_all.pkl')
print(f"Transformed shape: {X_train_trans.shape}")
