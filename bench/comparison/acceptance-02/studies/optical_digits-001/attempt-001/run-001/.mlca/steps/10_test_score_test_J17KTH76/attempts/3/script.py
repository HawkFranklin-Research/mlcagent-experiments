import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
primary = proto["task"]["primary_metric"]

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
test = pd.read_parquet(inp["test"])

excl = set(proto.get("targets") or []) | set(proto.get("excluded_columns") or [])
g = proto.get("group_column")
if g:
    excl.add(g)
features = [c for c in train.columns if c not in excl]

pre = load_pkl(inp["05_preprocess_fit_preprocessing_EJN3HZZN/preprocessor.pkl"])
sel = load_pkl(inp["06_features_retain_features_AY9YXAGS/selector_all.pkl"])

def make_ext(pre, sel):
    if pre is None:
        return sel
    if sel is None:
        return pre
    if isinstance(sel, Pipeline) and len(sel.steps) > 1:
        return sel
    return Pipeline([("pre", pre), ("sel", sel)])

ext = make_ext(pre, sel)

def final_est(model):
    return model.steps[-1][1] if isinstance(model, Pipeline) else model

def front_for(model, ext):
    if isinstance(model, Pipeline):
        if len(model.steps) > 1:
            return Pipeline(model.steps[:-1])
        return ext
    return ext

def transform(front, X):
    return X if front is None else front.transform(X)

def recon_mse(model, X, ext):
    front = front_for(model, ext)
    est = final_est(model)
    Z = transform(front, X)
    if est.__class__.__name__ == "PCA":
        rec = est.inverse_transform(est.transform(Z))
    elif est.__class__.__name__ == "MLPRegressor":
        rec = est.predict(Z)
    else:
        if hasattr(est, "inverse_transform"):
            rec = est.inverse_transform(est.transform(Z))
        else:
            rec = est.predict(Z)
    Za = np.asarray(Z, dtype=float)
    Ra = np.asarray(rec, dtype=float)
    return float(np.mean((Za - Ra) ** 2))

def act(name, A):
    if name == "relu":
        return np.maximum(A, 0.0)
    if name == "tanh":
        return np.tanh(A)
    if name == "logistic":
        return 1.0 / (1.0 + np.exp(-np.clip(A, -50.0, 50.0)))
    if name == "identity":
        return A
    raise ValueError("Unsupported activation: " + str(name))

def mlp_bottleneck(est, Z):
    A = np.asarray(Z, dtype=float)
    mid = len(est.hidden_layer_sizes) // 2
    for i in range(mid + 1):
        A = A @ np.asarray(est.coefs_[i], dtype=float) + np.asarray(est.intercepts_[i], dtype=float)
        A = act(est.activation, A)
    if A.ndim != 2 or not np.all(np.isfinite(A)):
        raise ValueError("Non-finite MLP bottleneck")
    return A

def embedding(model, X, ext):
    front = front_for(model, ext)
    est = final_est(model)
    Z = transform(front, X)
    if est.__class__.__name__ == "PCA":
        E = est.transform(Z)
    elif est.__class__.__name__ == "MLPRegressor":
        E = mlp_bottleneck(est, Z)
    else:
        raise ValueError("Unsupported embedding estimator: " + est.__class__.__name__)
    E = np.asarray(E, dtype=float)
    if E.ndim != 2 or not np.all(np.isfinite(E)):
        raise ValueError("Non-finite embedding")
    return E

cands = []
for label, key in [("baseline", "07_model_train_baseline_KG9SJ6DB/model.pkl"),
                   ("alternative", "08_model_train_alternative_WEEPWNFD/alternative_model.pkl")]:
    m = load_pkl(inp[key])
    name = label + "_" + final_est(m).__class__.__name__
    cands.append((name, m, recon_mse(m, val[features], ext)))

selected_name, selected_model, selected_val = min(cands, key=lambda t: t[2])
test_mse = recon_mse(selected_model, test[features], ext)

metrics = pd.DataFrame([{"model": selected_name, "metric": primary, "value": round(test_mse, 6)}])
mlca_runtime.save_table(metrics, "test_metrics.csv", caption="Test reconstruction error")
mlca_runtime.record_metric(primary, test_mse, split="test")
mlca_runtime.record_fact_candidate("test_score", test_mse, source="test_metrics.csv")

def emb_frame(df, part):
    E = embedding(selected_model, df[features], ext)
    out = pd.DataFrame(E, columns=[f"embedding_{i}" for i in range(E.shape[1])])
    out.insert(0, "row_id", df.index.to_numpy())
    out.insert(1, "partition", part)
    return out

all_emb = pd.concat([emb_frame(train, "train"), emb_frame(test, "test")], ignore_index=True)
mlca_runtime.save_row_outputs(all_emb, "embeddings.parquet")

print("rows train/val/test", len(train), len(val), len(test))
print("features", len(features))
print("selected", selected_name, "val_mse", round(float(selected_val), 6))
print("test_mse", round(float(test_mse), 6))
print("embeddings", all_emb.shape)
