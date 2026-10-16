import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

proto = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns", "group_column", "task"):
    if k not in proto:
        raise ValueError(f"missing {k}; available: {list(proto)}")
targets = proto["targets"]
excluded = proto["excluded_columns"]
group_col = proto["group_column"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def is_keylike(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s):
        u = np.sort(s.dropna().unique())
        if len(u) == len(s) and len(u) > 1 and (u == np.arange(len(u))).all():
            return True
    return False

feat_cols = [c for c in train.columns
             if c not in targets and c not in excluded and c != group_col
             and not is_keylike(train[c])]
X_train = train[feat_cols]
X_val = val[feat_cols]

with open(inp["05_preprocess_fit_preprocessing_RDQ93D4J/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
if hasattr(pre, "set_output"):
    pre.set_output(transform="pandas")

Xt = pre.transform(X_train)

pipe = Pipeline([
    ("pre", pre),
    ("mlp", MLPRegressor(hidden_layer_sizes=(16, 8, 16),
                         max_iter=300, random_state=42)),
])
mlca_runtime.fit_record(pipe, X_train, partition="train", y=Xt)
mlca_runtime.save_model(pipe, "alternative_model.pkl")

Xt_val = np.asarray(pre.transform(X_val))
rec = np.asarray(pipe.predict(X_val))
mse = float(np.mean((rec - Xt_val) ** 2))
mlca_runtime.record_metric("reconstruction_error", mse, "val")
print("features", len(feat_cols), "train", len(X_train), "val_mse", round(mse, 4))
