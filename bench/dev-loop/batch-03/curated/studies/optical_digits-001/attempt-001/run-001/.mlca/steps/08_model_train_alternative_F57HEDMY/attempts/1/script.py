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
targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def feature_columns(df):
    cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]
    feats = []
    for c in cols:
        s = df[c]
        if s.dtype == object:
            continue
        if s.dtype.kind in "iu" and s.nunique(dropna=True) == len(s):
            continue
        feats.append(c)
    return feats

feat_cols = feature_columns(train)

with open(inputs["05_preprocess_fit_preprocessing_NFXS64E4/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_retain_features_HGAM38DP/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)

def input_columns(obj, df, default):
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
    return default

if isinstance(selector, Pipeline):
    sel_cols = input_columns(selector, train, None)
    if sel_cols is not None:
        frontend = selector
        input_cols = sel_cols
    else:
        frontend = Pipeline([("prep", preprocessor), ("sel", selector)])
        input_cols = input_columns(preprocessor, train, feat_cols)
else:
    frontend = Pipeline([("prep", preprocessor), ("sel", selector)])
    input_cols = input_columns(preprocessor, train, feat_cols)

X_train = train[input_cols].copy()
X_val = val[input_cols].copy()

y_train = frontend.transform(X_train)

mlp = MLPRegressor(hidden_layer_sizes=(16, 8, 16), max_iter=300, random_state=42)
pipe = Pipeline([("frontend", frontend), ("mlp", mlp)])
mlca_runtime.fit_record(pipe, X_train, partition="train", y=y_train)

mlca_runtime.save_model(pipe, "alternative_model.pkl")

Z_train = pipe.named_steps["frontend"].transform(X_train)
Z_val = pipe.named_steps["frontend"].transform(X_val)
pred_train = pipe.predict(X_train)
pred_val = pipe.predict(X_val)
mse_train = float(np.mean((pred_train - Z_train) ** 2))
mse_val = float(np.mean((pred_val - Z_val) ** 2))
mlca_runtime.record_metric("reconstruction_error", mse_train, "train")
mlca_runtime.record_metric("reconstruction_error", mse_val, "val")

fitted_mlp = pipe.named_steps["mlp"]

def activate(x, act):
    if act == "relu":
        return np.maximum(0, x)
    if act == "tanh":
        return np.tanh(x)
    if act == "logistic":
        return 1 / (1 + np.exp(-x))
    return x

def embed(Z):
    act = fitted_mlp.activation
    h1 = activate(Z @ fitted_mlp.coefs_[0] + fitted_mlp.intercepts_[0], act)
    h2 = activate(h1 @ fitted_mlp.coefs_[1] + fitted_mlp.intercepts_[1], act)
    return h2

emb_train = embed(Z_train)
emb_val = embed(Z_val)
dim = emb_train.shape[1]
mlca_runtime.record_metric("embedding_dimension", dim, "train")

cols = [f"embedding_{i}" for i in range(dim)]
df_train = pd.DataFrame(emb_train, columns=cols, index=X_train.index)
df_train.index.name = "row_id"
df_val = pd.DataFrame(emb_val, columns=cols, index=X_val.index)
df_val.index.name = "row_id"

mlca_runtime.save_row_outputs(df_train.reset_index(), "alternative_embedding_train.parquet")
mlca_runtime.save_row_outputs(df_val.reset_index(), "alternative_embedding_val.parquet")

print(f"Train rows: {len(X_train)}, Val rows: {len(X_val)}, Embedding dim: {dim}")
print(f"Train reconstruction error: {mse_train:.4g}")
print(f"Val reconstruction error: {mse_val:.4g}")
