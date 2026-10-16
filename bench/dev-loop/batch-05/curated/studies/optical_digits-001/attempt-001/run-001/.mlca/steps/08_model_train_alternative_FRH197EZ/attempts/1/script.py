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
task = protocol["task"]
targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column", None)

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
with open(inputs["05_preprocess_fit_preprocessing_M7FAZM1F/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_retain_features_271P20QZ/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)

drop = set(targets) | set(excluded) | ({group_col} if group_col else set())

def key_like(s):
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_float_dtype(s):
        return False
    if len(s) <= 1:
        return False
    return s.nunique(dropna=False) == len(s)

feature_cols = [c for c in train.columns if c not in drop and not key_like(train[c])]

def input_names(obj):
    names = getattr(obj, "feature_names_in_", None)
    if names is not None:
        return list(names)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            n = input_names(step)
            if n is not None:
                return n
    return None

sel_in = input_names(selector)
if sel_in is not None and set(sel_in) == set(feature_cols):
    frontend = selector
else:
    frontend = Pipeline([("preprocessor", preprocessor), ("selector", selector)])

def to_df(a, index, cols):
    if isinstance(a, pd.DataFrame):
        return a
    a = np.asarray(a)
    if cols is not None and len(cols) == a.shape[1]:
        return pd.DataFrame(a, index=index, columns=cols)
    return pd.DataFrame(a, index=index, columns=[f"f{i}" for i in range(a.shape[1])])

Xraw = train[feature_cols]
processed = to_df(frontend.transform(Xraw), Xraw.index, None)
scaler = StandardScaler().set_output(transform="pandas").fit(processed)
scaled = to_df(scaler.transform(processed), processed.index, processed.columns)

mlp = MLPRegressor(hidden_layer_sizes=(16, 8, 16), max_iter=300, random_state=42)
fitted = mlca_runtime.fit_record(mlp, scaled, "train", y=scaled)

complete = Pipeline([("frontend", frontend), ("scaler", scaler), ("mlp", fitted)])
mlca_runtime.save_model(complete, "alternative_model.pkl")

sv = scaled.values
err = float(np.mean((fitted.predict(sv) - sv) ** 2))
mlca_runtime.record_metric("reconstruction_error", err, "train")

W1, W2, W3 = fitted.coefs_
b1, b2, b3 = fitted.intercepts_

def embed(A):
    h1 = np.maximum(A @ W1 + b1, 0.0)
    return np.maximum(h1 @ W2 + b2, 0.0)

def row_out(part_df, part_name):
    X = part_df[feature_cols]
    p = to_df(frontend.transform(X), X.index, processed.columns)
    s = to_df(scaler.transform(p), X.index, processed.columns)
    e = embed(s.values)
    df = pd.DataFrame(e, index=part_df.index, columns=[f"emb_{i}" for i in range(e.shape[1])])
    df.insert(0, "row_id", part_df.index)
    df.insert(1, "partition", part_name)
    return df

mlca_runtime.save_row_outputs(row_out(train, "train"), "alternative_embeddings_train.parquet")
mlca_runtime.save_row_outputs(row_out(val, "val"), "alternative_embeddings_val.parquet")

print("features:", len(feature_cols))
print("processed shape:", tuple(processed.shape))
print("embedding dim:", embed(sv).shape[1])
print("recon_mse_scaled:", round(err, 4))
