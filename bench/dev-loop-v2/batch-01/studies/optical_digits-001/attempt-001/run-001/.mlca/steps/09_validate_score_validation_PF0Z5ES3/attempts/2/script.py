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
targets = proto.get("targets") or []
excluded = proto.get("excluded_columns") or []
group = proto.get("group_column")
meta = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in meta]
for c in list(feature_cols):
    if not pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique(dropna=True) == len(train):
        feature_cols.remove(c)
X_val = val[list(feature_cols)].copy()

def load(name):
    with open(paths[name], 'rb') as f:
        return pickle.load(f)

prep = load("05_preprocess_fit_preprocessing_KTEB41J1/preprocessor.pkl")
sel = load("06_features_retain_features_K0P2E6DF/selector_all.pkl")
base = load("07_model_train_baseline_Y4Q85156/model.pkl")
alt = load("08_model_train_alternative_CDFZSS39/alternative_model.pkl")

def front_est(m):
    if isinstance(m, Pipeline) and len(m.named_steps) >= 2:
        return m[:-1], m[-1]
    return None, m

def default_front(X):
    for step in (prep, sel):
        try:
            X = step.transform(X)
        except Exception:
            X = step.transform(np.asarray(X, dtype=float))
    return X

def recon(est, Z):
    Za = np.asarray(Z, dtype=float)
    if isinstance(est, PCA):
        R = est.inverse_transform(est.transform(Za))
    elif isinstance(est, MLPRegressor):
        R = est.predict(Za)
    elif hasattr(est, 'inverse_transform') and hasattr(est, 'transform'):
        R = est.inverse_transform(est.transform(Za))
    elif hasattr(est, 'predict'):
        R = est.predict(Za)
    else:
        raise ValueError(f"Unsupported estimator: {type(est).__name__}")
    Ra = np.asarray(R, dtype=float)
    return float(np.mean((Za - Ra) ** 2))

def score(m):
    front, est = front_est(m)
    Z = front.transform(X_val) if front is not None else default_front(X_val)
    return recon(est, Z)

results = [("baseline", score(base)), ("alternative", score(alt))]
df = pd.DataFrame([{"model": n, "metric": primary, "value": float(f"{v:.4g}")} for n, v in results])
mlca_runtime.save_table(df, "validation_metrics.csv", caption="Validation reconstruction errors per candidate")

best_idx = int(np.argmin([v for _, v in results]))
best_name, best_score = results[best_idx][0], float(results[best_idx][1])
mlca_runtime.record_metric(primary, best_score, "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", float(f"{best_score:.6g}"), source="validation_metrics.csv")

print(f"features={len(feature_cols)}, val_rows={len(X_val)}")
print(f"best={best_name} {primary}={best_score:.6g}")
