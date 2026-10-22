import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())
primary = proto["task"]["primary_metric"]

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

targets = proto["targets"] or []
excluded = proto["excluded_columns"] or []
group = proto.get("group_column")
meta = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in meta]
for c in list(feature_cols):
    if not pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique(dropna=True) == len(train):
        feature_cols.remove(c)

X_val = val[feature_cols].copy()

with open(paths["05_preprocess_fit_preprocessing_KTEB41J1/preprocessor.pkl"], 'rb') as f:
    prep = pickle.load(f)
with open(paths["06_features_retain_features_K0P2E6DF/selector_all.pkl"], 'rb') as f:
    sel = pickle.load(f)
with open(paths["07_model_train_baseline_Y4Q85156/model.pkl"], 'rb') as f:
    base = pickle.load(f)
with open(paths["08_model_train_alternative_CDFZSS39/alternative_model.pkl"], 'rb') as f:
    alt = pickle.load(f)

def in_cols(o):
    return list(o.feature_names_in_) if hasattr(o, 'feature_names_in_') else None

def safe_transform(obj, X):
    if isinstance(X, pd.DataFrame) and hasattr(obj, 'feature_names_in_'):
        req = list(obj.feature_names_in_)
        if set(req).issubset(set(X.columns)):
            X = X[req]
    return obj.transform(X)

def to_df(out, obj, index):
    if isinstance(out, pd.DataFrame):
        return out
    try:
        cols = list(obj.get_feature_names_out())
    except Exception:
        cols = None
    if cols is not None and getattr(out, 'shape', (0, 0))[1] == len(cols):
        return pd.DataFrame(out, index=index, columns=cols)
    return pd.DataFrame(out, index=index)

def final_est(m):
    return m.named_steps[list(m.named_steps.keys())[-1]] if isinstance(m, Pipeline) else m

def plen(o):
    return len(o.named_steps) if isinstance(o, Pipeline) else 1

raw = list(feature_cols)
prep_in, sel_in = in_cols(prep), in_cols(sel)
if prep_in and set(prep_in) == set(raw) and sel_in and set(sel_in) == set(raw):
    chain = [sel] if plen(sel) >= plen(prep) else [prep]
else:
    chain = [prep, sel]

Z = X_val
for step in chain:
    Z = safe_transform(step, Z)
    Z = to_df(Z, step, X_val.index)

def recon_mse(f_est, Z):
    if isinstance(f_est, PCA):
        Zr = f_est.inverse_transform(f_est.transform(Z))
    elif isinstance(f_est, MLPRegressor):
        Zr = f_est.predict(Z)
    elif hasattr(f_est, 'inverse_transform') and hasattr(f_est, 'transform'):
        Zr = f_est.inverse_transform(f_est.transform(Z))
    elif hasattr(f_est, 'predict'):
        Zr = f_est.predict(Z)
    else:
        raise ValueError(f"Unsupported estimator: {type(f_est).__name__}")
    Za = np.asarray(Z, dtype=float); Zra = np.asarray(Zr, dtype=float)
    return float(np.mean((Za - Zra) ** 2))

candidates = [("baseline", final_est(base)), ("alternative", final_est(alt))]
results = [(name, primary, recon_mse(f_est, Z)) for name, f_est in candidates]

df = pd.DataFrame(results, columns=["model", "metric", "value"])
df["value"] = df["value"].apply(lambda v: float(f"{v:.4g}"))
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation reconstruction errors per candidate")

best_name, _, best_score = min(results, key=lambda r: r[2])
mlca_runtime.record_metric(primary, best_score, "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", float(f"{best_score:.6g}"), source="validation_metrics.csv")

print(f"features={len(feature_cols)}, val_rows={len(X_val)}, frontend_dim={Z.shape[1]}")
print(f"best={best_name} {primary}={best_score:.6g}")
