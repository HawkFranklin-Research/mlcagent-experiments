import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

def load_pkl(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

protocol = json.loads(inp["_protocol"].read_text())
try:
    primary_metric = protocol["task"]["primary_metric"]
    targets = list(protocol["targets"])
    excluded = list(protocol["excluded_columns"])
except KeyError as e:
    raise KeyError(f"Missing {e}; available top-level keys: {list(protocol)}")
group = protocol.get("group_column")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)
feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if (pd.api.types.is_string_dtype(s) or s.dtype == object) and s.nunique(dropna=True) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
        continue
    feature_cols.append(c)

missing = [c for c in feature_cols if c not in val.columns]
if missing:
    raise ValueError(f"Validation missing feature columns: {missing[:5]}")
Xval = val[feature_cols].copy()

pre = load_pkl("05_preprocess_fit_preprocessing_EJN3HZZN/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_AY9YXAGS/selector_all.pkl")
base = load_pkl("07_model_train_baseline_KG9SJ6DB/model.pkl")
alt = load_pkl("08_model_train_alternative_WEEPWNFD/alternative_model.pkl")

def subset_for(obj, X):
    if hasattr(obj, "feature_names_in_"):
        cols = list(obj.feature_names_in_)
        if all(c in X.columns for c in cols):
            return X[cols]
        raise ValueError(f"Missing columns for {type(obj).__name__}: {[c for c in cols if c not in X.columns][:3]}")
    return X

def try_apply(obj, X):
    if obj is None or not hasattr(obj, "transform"):
        raise ValueError("no transform")
    return obj.transform(subset_for(obj, X))

def front_transform(X):
    try:
        return try_apply(sel, X)
    except Exception:
        pass
    try:
        Z = try_apply(pre, X)
        if hasattr(sel, "feature_names_in_") and not isinstance(Z, pd.DataFrame):
            Z = pd.DataFrame(Z, columns=list(sel.feature_names_in_), index=X.index)
        return sel.transform(subset_for(sel, Z) if isinstance(Z, pd.DataFrame) else Z)
    except Exception:
        pass
    try:
        return try_apply(pre, X)
    except Exception:
        return X[feature_cols]

def get_final_front(model, X):
    if isinstance(model, Pipeline):
        final = model.steps[-1][1]
        if len(model.steps) <= 1:
            return final, X
        front = model[:-1]
        return final, front.transform(subset_for(front, X))
    return model, front_transform(X)

def prep_final(final, Xf, index):
    if hasattr(final, "feature_names_in_"):
        cols = list(final.feature_names_in_)
        if isinstance(Xf, pd.DataFrame):
            if all(c in Xf.columns for c in cols):
                return Xf[cols]
        else:
            arr = np.asarray(Xf)
            if arr.ndim == 2 and arr.shape[1] == len(cols):
                return pd.DataFrame(arr, columns=cols, index=index)
    return Xf

def recon_mse(final, Xf, index):
    Xp = prep_final(final, Xf, index)
    if isinstance(final, PCA) or (hasattr(final, "inverse_transform") and hasattr(final, "components_")):
        R = final.inverse_transform(final.transform(Xp))
    elif isinstance(final, MLPRegressor) or (hasattr(final, "coefs_") and hasattr(final, "predict")):
        R = final.predict(Xp)
    else:
        raise TypeError(f"Unsupported final estimator: {type(final)}")
    return float(np.mean((np.asarray(Xp, dtype=float) - np.asarray(R, dtype=float)) ** 2))

def cand_name(final):
    if isinstance(final, PCA):
        return "PCA"
    if isinstance(final, MLPRegressor):
        return "MLPRegressor"
    return type(final).__name__

rows = []
for model in [base, alt]:
    final, Xf = get_final_front(model, Xval)
    rows.append({"model": cand_name(final), "metric": primary_metric, "value": recon_mse(final, Xf, Xval.index)})

df = pd.DataFrame(rows)
best_idx = df["value"].idxmin()
best_model = str(df.loc[best_idx, "model"])
best_score = float(df.loc[best_idx, "value"])
out = df.copy()
out["value"] = out["value"].map(lambda v: float(f"{v:.4g}"))
mlca_runtime.save_table(out, "validation_metrics.csv", caption="Validation reconstruction error in fitted frontend feature space.")
mlca_runtime.record_metric(primary_metric, best_score, "val")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")
print(f"features={len(feature_cols)} val_rows={len(Xval)}")
print(f"best={best_model} {primary_metric}={best_score:.6g}")
