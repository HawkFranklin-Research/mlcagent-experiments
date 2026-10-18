import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available: {list(protocol)}")
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; available task keys: {list(task)}")
metric = task["primary_metric"]
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
drop = set(targets) | set(excluded)
if group_col is not None:
    drop.add(group_col)

feat_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if (pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)) and s.nunique(dropna=False) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=False) == len(s):
        continue
    feat_cols.append(c)
if not feat_cols:
    raise ValueError("No feature columns found")
Xval = val[feat_cols].copy()

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inp["05_preprocess_fit_preprocessing_JYNCWEGN/preprocessor.pkl"])
sel = load_pkl(inp["06_features_retain_features_HY561TZJ/selector_all.pkl"])

def transform(obj, X):
    if obj is None:
        return X
    if isinstance(obj, Pipeline):
        return obj.transform(X)
    if hasattr(obj, "transform"):
        return obj.transform(X)
    raise TypeError(f"Object {type(obj).__name__} has no transform")

def split_candidate(obj):
    if isinstance(obj, Pipeline) and obj.steps:
        final = obj.steps[-1][1]
        front_steps = obj.steps[:-1]
        front = Pipeline(front_steps) if front_steps else None
        return front, final
    return None, obj

def external_options():
    if isinstance(sel, Pipeline):
        opts = [sel]
        if pre is not None:
            opts.append(Pipeline([("pre", pre), ("sel", sel)]))
        return opts
    opts = []
    if pre is not None and sel is not None:
        opts.append(Pipeline([("pre", pre), ("sel", sel)]))
    if sel is not None:
        opts.append(sel)
    if pre is not None:
        opts.append(pre)
    return opts

def frontend_for(obj):
    front, final = split_candidate(obj)
    sample = Xval.head(min(5, len(Xval)))
    nfeat = getattr(final, "n_features_in_", None)
    if front is not None:
        try:
            Z = np.asarray(transform(front, sample))
            if Z.ndim == 2 and (nfeat is None or Z.shape[1] == nfeat):
                return front, final
        except Exception:
            pass
    for opt in external_options():
        try:
            Z = np.asarray(transform(opt, sample))
        except Exception:
            continue
        if Z.ndim == 2 and (nfeat is None or Z.shape[1] == nfeat):
            return opt, final
    if nfeat is None or nfeat == len(feat_cols):
        return None, final
    raise ValueError(f"Cannot build frontend for {type(final).__name__}")

def reconstruct(final, Xf):
    if isinstance(final, PCA):
        return final.inverse_transform(final.transform(Xf))
    if isinstance(final, MLPRegressor):
        return final.predict(Xf)
    if hasattr(final, "predict"):
        return final.predict(Xf)
    if hasattr(final, "transform") and hasattr(final, "inverse_transform"):
        return final.inverse_transform(final.transform(Xf))
    raise TypeError(f"Cannot reconstruct with {type(final).__name__}")

paths = {
    "baseline": inp["07_model_train_baseline_V6BDRB4H/model.pkl"],
    "alternative": inp["08_model_train_alternative_WZMYQEJM/alternative_model.pkl"],
}
rows = []
for name, path in paths.items():
    obj = load_pkl(path)
    front, final = frontend_for(obj)
    Xf = transform(front, Xval) if front is not None else Xval
    Xfa = np.asarray(Xf, dtype=float)
    if Xfa.ndim != 2:
        raise ValueError("Frontend output is not 2D")
    nfeat = getattr(final, "n_features_in_", None)
    if nfeat is not None and Xfa.shape[1] != nfeat:
        raise ValueError(f"{name}: frontend width {Xfa.shape[1]} != estimator width {nfeat}")
    R = np.asarray(reconstruct(final, Xf), dtype=float)
    if R.shape != Xfa.shape:
        raise ValueError(f"{name}: reconstruction shape {R.shape} != input shape {Xfa.shape}")
    mse = float(np.mean((Xfa - R) ** 2))
    if not np.isfinite(mse):
        raise ValueError(f"{name}: non-finite reconstruction error")
    rows.append({"model": name, "metric": metric, "value": float(f"{mse:.4g}")})

out = pd.DataFrame(rows, columns=["model", "metric", "value"])
mlca_runtime.save_table(out, "validation_metrics.csv", caption="Validation reconstruction error by candidate")
lower_is_better = any(s in metric.lower() for s in ("error", "loss", "rmse", "mae", "mse"))
best = out.sort_values("value", ascending=lower_is_better).iloc[0]
best_name = str(best["model"])
best_value = float(best["value"])
mlca_runtime.record_metric(metric, best_value, "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", best_value, source="validation_metrics.csv")
print(f"rows train={len(train)} val={len(val)} features={len(feat_cols)}")
print(f"best={best_name} {metric}={best_value:.4g}")
