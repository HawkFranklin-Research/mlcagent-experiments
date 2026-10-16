import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_M7FAZM1F/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_retain_features_271P20QZ/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)

def key_like(s):
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_float_dtype(s):
        return False
    if len(s) <= 1:
        return False
    return s.nunique(dropna=False) == len(s)

feature_cols = [c for c in train.columns if c not in drop and not key_like(train[c])]

def input_names(obj):
    n = getattr(obj, "feature_names_in_", None)
    if n is not None:
        return list(n)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            r = input_names(step)
            if r is not None:
                return r
    return None

sel_in = input_names(selector)
if sel_in is not None and set(sel_in) == set(feature_cols):
    frontend = selector
else:
    frontend = Pipeline([("preprocessor", preprocessor), ("selector", selector)])

Xraw = train[feature_cols]
processed = frontend.transform(Xraw)
if hasattr(processed, "toarray"):
    processed = processed.toarray()
if not isinstance(processed, pd.DataFrame):
    processed = pd.DataFrame(np.asarray(processed), index=Xraw.index)

inner = Pipeline([
    ("scaler", StandardScaler().set_output(transform="pandas")),
    ("mlp", MLPRegressor(hidden_layer_sizes=(16, 8, 16), max_iter=300, random_state=42)),
])
fitted = mlca_runtime.fit_record(inner, processed, "train", y=processed)

complete = Pipeline([("frontend", frontend), ("model", fitted)])
mlca_runtime.save_model(complete, "alternative_model.pkl")

scaler = fitted.named_steps["scaler"]
mlp = fitted.named_steps["mlp"]
sv = scaler.transform(processed)
if hasattr(sv, "toarray"):
    sv = sv.toarray()
if not isinstance(sv, pd.DataFrame):
    sv = pd.DataFrame(np.asarray(sv), index=processed.index)
svv = np.asarray(sv)
err = float(np.mean((mlp.predict(svv) - svv) ** 2))
mlca_runtime.record_metric("reconstruction_error", err, "train")

def embed(A):
    h = np.asarray(A)
    h = np.maximum(h @ mlp.coefs_[0] + mlp.intercepts_[0], 0.0)
    h = np.maximum(h @ mlp.coefs_[1] + mlp.intercepts_[1], 0.0)
    return h

emb = embed(svv)
print("features:", len(feature_cols))
print("processed shape:", tuple(processed.shape))
print("embedding dim:", emb.shape[1])
print("recon_mse_scaled:", round(err, 4))
