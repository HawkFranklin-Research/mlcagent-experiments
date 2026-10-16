import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for k in ("targets", "excluded_columns"):
    if k not in protocol:
        raise KeyError(f"Missing {k} in protocol. Available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])

def is_key_like(s):
    if s.dtype == object:
        return True
    if s.dtype.kind in "iu" and s.nunique(dropna=True) == len(s):
        return True
    return False

feat_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col and not is_key_like(train[c])]

with open(inputs["05_preprocess_fit_preprocessing_NFXS64E4/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_retain_features_HGAM38DP/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)

def get_input_cols(obj, df):
    if hasattr(obj, "feature_names_in_"):
        cols = list(obj.feature_names_in_)
        if all(c in df.columns for c in cols):
            return cols
    if isinstance(obj, Pipeline):
        first = obj.steps[0][1]
        if hasattr(first, "feature_names_in_"):
            cols = list(first.feature_names_in_)
            if all(c in df.columns for c in cols):
                return cols
    return None

sel_cols = get_input_cols(selector, train)
if sel_cols is not None:
    frontend = selector
    input_cols = sel_cols
else:
    frontend = Pipeline([("prep", preprocessor), ("sel", selector)])
    input_cols = get_input_cols(preprocessor, train) or feat_cols

input_cols = [c for c in input_cols if c not in targets and c not in excluded and c != group_col]
X_train = train[input_cols].copy()
y_train = frontend.transform(X_train)
if hasattr(y_train, "toarray"):
    y_train = y_train.toarray()
else:
    y_train = np.asarray(y_train)

mlp = MLPRegressor(hidden_layer_sizes=(16, 8, 16), max_iter=300, random_state=42)
pipe = Pipeline([("frontend", frontend), ("mlp", mlp)])
mlca_runtime.fit_record(pipe, X_train, partition="train", y=y_train)

mlca_runtime.save_model(pipe, "alternative_model.pkl")

pred_train = pipe.predict(X_train)
mse_train = float(np.mean((pred_train - y_train) ** 2))
mlca_runtime.record_metric("reconstruction_error", mse_train, "train")
dim = pipe.named_steps["mlp"].hidden_layer_sizes[1]
mlca_runtime.record_metric("embedding_dimension", dim, "train")

print(f"Train rows: {len(X_train)}, features: {X_train.shape[1]}")
print(f"Processed features: {y_train.shape[1]}")
print(f"Train reconstruction error: {mse_train:.4g}")
