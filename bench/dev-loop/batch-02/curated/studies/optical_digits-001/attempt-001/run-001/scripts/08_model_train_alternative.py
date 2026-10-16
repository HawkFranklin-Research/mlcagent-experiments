import mlca_runtime
import pandas as pd
import json
import pickle
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_squared_error

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

def get_key(proto, key):
    if key not in proto:
        raise KeyError(f"Missing key '{key}'. Available: {list(proto.keys())}")
    return proto[key]

proto = json.loads(inputs['_protocol'].read_text())
targets = get_key(proto, 'targets')
excluded = get_key(proto, 'excluded_columns')
group_col = get_key(proto, 'group_column')
task = get_key(proto, 'task')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

def is_key_like(s):
    if s.dtype == 'object' or pd.api.types.is_string_dtype(s):
        return s.nunique() == len(s)
    return False

feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group_col
    and not is_key_like(train[c])
]

with open(inputs['05_preprocess_fit_preprocessing_78KYAZ91/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_retain_features_01V2VW4T/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)

def is_complete_frontend(sel, raw_cols):
    if isinstance(sel, Pipeline):
        if 'preprocessor' in sel.named_steps:
            return True
        if hasattr(sel, 'feature_names_in_') and set(sel.feature_names_in_).issubset(set(raw_cols)):
            return True
    return False

if is_complete_frontend(selector, feature_cols):
    frontend = selector
else:
    frontend = Pipeline([('preprocessor', preprocessor), ('selector', selector)])

X_train = train[feature_cols]
y_train = frontend.transform(X_train)

ft = FunctionTransformer(frontend.transform)
mlp = MLPRegressor(hidden_layer_sizes=(16, 8, 16), max_iter=300, random_state=42)
model = Pipeline([('frontend', ft), ('mlp', mlp)])

fitted_model = mlca_runtime.fit_record(model, X_train, partition='train', y=y_train)

X_val = val[feature_cols]
y_val = frontend.transform(X_val)
y_pred = fitted_model.predict(X_val)
val_mse = mean_squared_error(y_val, y_pred)
mlca_runtime.record_metric("reconstruction_error", float(val_mse), split="val")
mlca_runtime.record_metric("embedding_dimension", 8, split="train")

mlca_runtime.save_model(fitted_model, "alternative_model.pkl")

print(f"Train rows: {len(train)}, Val rows: {len(val)}")
print(f"Feature columns: {len(feature_cols)}")
print(f"Val reconstruction error (scaled space): {val_mse:.4f}")
