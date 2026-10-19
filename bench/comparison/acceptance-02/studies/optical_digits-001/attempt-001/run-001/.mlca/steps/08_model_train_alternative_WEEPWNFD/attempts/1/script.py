import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

# Load and validate protocol
protocol = json.loads(inputs['_protocol'].read_text())
required = ['targets', 'excluded_columns', 'group_column', 'task']
for k in required:
    if k not in protocol:
        raise KeyError(f"Missing key {k} in protocol. Available: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column', None)
task = protocol['task']

# Load data
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

# Load saved preprocessing objects
with open(inputs['05_preprocess_fit_preprocessing_EJN3HZZN/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_retain_features_AY9YXAGS/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)

# Determine feature columns from preprocessor contract
if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = [c for c in preprocessor.feature_names_in_ if c in train.columns]
    if len(feature_cols) != len(preprocessor.feature_names_in_):
        raise ValueError("Some feature columns from preprocessor not found in train")
else:
    feature_cols = [c for c in train.columns
                    if c not in targets + excluded + ([group] if group else [])]

X_raw = train[feature_cols].copy()

# Compute processed training features (reconstruction target)
y = preprocessor.transform(X_raw)
if selector is not None:
    y = selector.transform(y)

# Build complete pipeline
steps = [('pre', preprocessor)]
if selector is not None:
    steps.append(('sel', selector))
steps.append(('mlp', MLPRegressor(hidden_layer_sizes=(16, 8, 16),
                                  max_iter=300,
                                  random_state=42)))
pipeline = Pipeline(steps)

# Fit on original training DataFrame
fitted = mlca_runtime.fit_record(pipeline, X_raw, 'train', y=y)

# Save model
mlca_runtime.save_model(fitted, 'alternative_model.pkl')

# Evaluation helper
def recon_error(df):
    X = df[feature_cols].copy()
    Xt = fitted.named_steps['pre'].transform(X)
    if 'sel' in fitted.named_steps:
        Xt = fitted.named_steps['sel'].transform(Xt)
    pred = fitted.named_steps['mlp'].predict(Xt)
    return mean_squared_error(Xt, pred), Xt.shape[1]

train_err, n_processed = recon_error(train)
val_err, _ = recon_error(val)

mlca_runtime.record_metric('reconstruction_error', train_err, 'train')
mlca_runtime.record_metric('reconstruction_error', val_err, 'val')

print(f"Train shape: {train.shape}")
print(f"Feature count: {len(feature_cols)}")
print(f"Processed feature count: {n_processed}")
print(f"Train reconstruction error: {train_err:.4f}")
print(f"Val reconstruction error: {val_err:.4f}")
