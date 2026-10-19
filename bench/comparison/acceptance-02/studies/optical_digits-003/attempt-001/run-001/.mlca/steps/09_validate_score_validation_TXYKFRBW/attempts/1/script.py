import mlca_runtime as mlca
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

mlca.set_seed(42)

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def locate(obj, cls):
    if isinstance(obj, Pipeline):
        for i, (_, step) in enumerate(obj.steps):
            if isinstance(step, cls):
                return (Pipeline(obj.steps[:i]) if i > 0 else None), step
            if isinstance(step, Pipeline):
                pre, est = locate(step, cls)
                if est is not None:
                    steps = list(obj.steps[:i]) + (list(pre.steps) if pre is not None else [])
                    return (Pipeline(steps) if steps else None), est
        return None, None
    return (None, obj) if isinstance(obj, cls) else (None, None)

def expected_dim(est):
    if hasattr(est, "n_features_in_"):
        return int(est.n_features_in_)
    if isinstance(est, PCA) and hasattr(est, "components_"):
        return int(est.components_.shape[1])
    if isinstance(est, MLPRegressor) and hasattr(est, "coefs_"):
        return int(est.coefs_[0].shape[0])
    return None

def as_array(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a

def align(recon, target):
    r = as_array(recon)
    if r.shape != target.shape:
        if r.size == target.size:
            r = r.reshape(target.shape)
        else:
            raise ValueError(f"reconstruction shape {r.shape} != input {target.shape}")
    return r

def key_like(s):
    if len(s) == 0:
        return False
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and len(s) > 1:
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s) and len(s) > 1 and s.nunique(dropna=False) == len(s):
        vals = np.sort(s.dropna().unique())
        return len(vals) == len(s) and np.all(np.diff(vals) == 1)
    return False

inputs = mlca.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
primary = protocol["task"]["primary_metric"]
if primary != "reconstruction_error":
    raise ValueError(f"Declared primary metric is {primary!r}, expected 'reconstruction_error'")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group and not key_like(train[c])
]
val_features = val.loc[:, feature_cols].copy()

pre = load_pkl(inputs["05_preprocess_fit_preprocessing_4ZZYYY8Y/preprocessor.pkl"])
sel = load_pkl(inputs["06_features_retain_features_E2FAAE9F/selector_all.pkl"])
base = load_pkl(inputs["07_model_train_baseline_Q9V0XT19/model.pkl"])
alt = load_pkl(inputs["08_model_train_alternative_GWTD8WJA/alternative_model.pkl"])

base_fe, pca = locate(base, PCA)
alt_fe, mlp = locate(alt, MLPRegressor)
if pca is None or mlp is None:
    raise ValueError("Could not locate both PCA and MLPRegressor in saved models")

frontends = []
if base_fe is not None:
    frontends.append(("baseline_frontend", base_fe))
if alt_fe is not None:
    frontends.append(("alternative_frontend", alt_fe))
if isinstance(sel, Pipeline):
    frontends.append(("selector_all", sel))
elif pre is not None and sel is not None and hasattr(sel, "transform"):
    frontends.append(("preprocessor+selector", Pipeline([("preprocessor", pre), ("selector", sel)])))
elif pre is not None and hasattr(pre, "transform"):
    frontends.append(("preprocessor", pre))
frontends.append(("identity", None))

d_pca, d_mlp = expected_dim(pca), expected_dim(mlp)
chosen = None
for name, fe in frontends:
    try:
        x = val_features if fe is None else fe.transform(val_features)
        xa = as_array(x)
    except Exception:
        continue
    if d_pca is not None and xa.shape[1] != d_pca:
        continue
    if d_mlp is not None and xa.shape[1] != d_mlp:
        continue
    chosen = (name, xa)
    break
if chosen is None:
    raise ValueError(f"No frontend matches PCA({d_pca}) and MLP({d_mlp})")
front_name, Xf = chosen

pca_recon = align(pca.inverse_transform(pca.transform(Xf)), Xf)
mlp_recon = align(mlp.predict(Xf), Xf)
pca_mse = float(np.mean((pca_recon - Xf) ** 2))
mlp_mse = float(np.mean((mlp_recon - Xf) ** 2))

metrics = pd.DataFrame([
    {"model": type(pca).__name__, "metric": primary, "value": pca_mse},
    {"model": type(mlp).__name__, "metric": primary, "value": mlp_mse},
])
metrics["value"] = [float(f"{v:.4g}") for v in metrics["value"]]
mlca.save_table(metrics, "validation_metrics.csv", caption="Validation reconstruction error by candidate.")

best_idx = metrics["value"].idxmin()
best_model = str(metrics.loc[best_idx, "model"])
best_value = float(metrics.loc[best_idx, "value"])
mlca.record_metric(primary, best_value, split="val")
mlca.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
mlca.record_fact_candidate("best_validation_score", best_value, source="validation_metrics.csv")

print(f"val rows={len(val_features)}, features={len(feature_cols)}")
print(f"frontend={front_name}, frontend shape={Xf.shape}")
for row in metrics.itertuples(index=False):
    print(f"{row.model}: {row.value:.4g}")
print(f"selected={best_model}, score={best_value:.4g}")
