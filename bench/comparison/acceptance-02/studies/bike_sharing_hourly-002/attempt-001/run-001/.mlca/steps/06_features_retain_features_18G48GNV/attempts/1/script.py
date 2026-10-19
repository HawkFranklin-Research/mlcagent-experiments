import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
for key in ['task', 'targets', 'excluded_columns', 'group_column']:
    if key not in protocol:
        raise KeyError(f"Missing key {key} in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_7BD2DR19/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

if not hasattr(preprocessor, 'feature_names_in_'):
    raise AttributeError("Preprocessor missing feature_names_in_")
feature_cols = preprocessor.feature_names_in_.tolist()

for col in list(targets) + list(excluded) + [group_col]:
    if col in feature_cols:
        raise ValueError(f"Column {col} should not be in feature columns")

X_raw = train[feature_cols].copy()
X_pre = preprocessor.transform(X_raw)

if not isinstance(X_pre, pd.DataFrame):
    if hasattr(preprocessor, 'get_feature_names_out'):
        names = preprocessor.get_feature_names_out()
        X_pre = pd.DataFrame(X_pre, columns=names, index=X_raw.index)
    else:
        raise TypeError("Preprocessed output is not a DataFrame and cannot get feature names")

selector = FunctionTransformer(func=None, validate=False)
mlca_runtime.fit_record(selector, X_pre, partition='train')

mlca_runtime.save_model(selector, 'selector_all.pkl')

print(f"Raw feature columns: {len(feature_cols)}")
print(f"Preprocessed shape: {X_pre.shape}")
print("Passthrough selector fitted and saved as selector_all.pkl")
