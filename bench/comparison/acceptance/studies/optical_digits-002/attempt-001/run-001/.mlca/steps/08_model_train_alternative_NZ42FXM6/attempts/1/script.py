import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {list(d)}")
    return d[k]

targets = list(req(proto, "targets"))
excluded = list(req(proto, "excluded_columns"))
task = req(proto, "task")
group_col = proto.get("group_column", None)
primary_metric = task.get("primary_metric", "reconstruction_error")

train = pd.read_parquet(inp["train"])

def key_like(s):
    n = int(s.notna().sum())
    if n == 0 or s.nunique(dropna=True) != n:
        return False
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        return True
    if pd.api.types.is_integer_dtype(s.dtype):
        vals = np.sort(s.dropna().unique())
        return len(vals) > 1 and np.all(np.diff(vals) == 1)
    return False

blocked = set(targets) | set(excluded)
if group_col is not None:
    blocked.add(group_col)
feature_cols = [c for c in train.columns if c not in blocked and not key_like(train[c])]
if not feature_cols:
    raise ValueError("No feature columns after applying the declared feature contract")
X = train[feature_cols].copy()

with open(inp["05_preprocess_fit_preprocessing_XD81KYMB/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_retain_features_75C8096C/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

preproc_types = (ColumnTransformer, SimpleImputer, StandardScaler, OneHotEncoder)
def contains_pre(pipe):
    return isinstance(pipe, Pipeline) and any(
        isinstance(st, preproc_types) or "preprocess" in name.lower()
        for name, st in pipe.named_steps.items()
    )

if contains_pre(sel):
    frontend = sel
else:
    frontend = Pipeline([("preprocess", pre), ("selector", sel)])

mlp = MLPRegressor(
    hidden_layer_sizes=(16, 8, 16),
    max_iter=300,
    random_state=42,
)
pipe = Pipeline([("frontend", frontend), ("mlp", mlp)])

Z = frontend.transform(X)
Z_width = int(Z.shape[1])
if Z_width <= 300 and hasattr(frontend, "set_output"):
    try:
        frontend.set_output(transform="pandas")
    except Exception:
        pass
    Z = frontend.transform(X)
if hasattr(Z, "toarray"):
    Z = Z.toarray()
Z = np.asarray(Z, dtype=float)

pipe = mlca_runtime.fit_record(pipe, X, "train", y=Z)

Z_fit = pipe.named_steps["frontend"].transform(X)
if hasattr(Z_fit, "toarray"):
    Z_fit = Z_fit.toarray()
Z_fit = np.asarray(Z_fit, dtype=float)
pred = pipe.named_steps["mlp"].predict(Z_fit)
mse = float(np.mean((pred - Z_fit) ** 2))

mlca_runtime.record_metric(primary_metric, mse, "train")
mlca_runtime.record_metric("embedding_dim", 8, "train")
mlca_runtime.save_model(pipe, "alternative_model.pkl")

print(f"features={len(feature_cols)} processed={Z_width}")
print(f"train reconstruction MSE in frontend feature space={mse:.4g}")
