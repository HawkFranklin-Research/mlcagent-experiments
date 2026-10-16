import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
task = protocol["task"]
primary_metric = task["primary_metric"]
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_keylike(s):
    if s.dtype == object:
        return s.nunique() == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique() == len(s)
    return False

feat_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col and not is_keylike(train[c])]
X_val = val[feat_cols]

with open(inputs["05_preprocess_fit_preprocessing_M7FAZM1F/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_271P20QZ/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)
with open(inputs["07_model_train_baseline_DR0KT0NW/model.pkl"], "rb") as f:
    base = pickle.load(f)
with open(inputs["08_model_train_alternative_FRH197EZ/alternative_model.pkl"], "rb") as f:
    alt = pickle.load(f)

def get_frontend(pre, sel):
    if isinstance(sel, Pipeline):
        return sel
    return Pipeline([("pre", pre), ("sel", sel)])

def get_scaled(X_raw, model, pre, sel):
    if isinstance(model, Pipeline) and len(model.steps) > 1:
        return Pipeline(model.steps[:-1]).transform(X_raw)
    return get_frontend(pre, sel).transform(X_raw)

def get_recon(model, X_raw, pre, sel):
    if isinstance(model, Pipeline):
        return model.predict(X_raw)
    return model.predict(get_frontend(pre, sel).transform(X_raw))

def recon_error(model, X_raw, pre, sel):
    Xs = get_scaled(X_raw, model, pre, sel)
    Xr = get_recon(model, X_raw, pre, sel)
    return float(np.mean((Xs - Xr) ** 2))

err_base = recon_error(base, X_val, pre, sel)
err_alt = recon_error(alt, X_val, pre, sel)

results = pd.DataFrame({
    "model": ["baseline", "alternative"],
    "metric": [primary_metric, primary_metric],
    "value": [err_base, err_alt]
})
mlca_runtime.save_table(results, "validation_metrics.csv", caption="Validation reconstruction error")

best_idx = int(np.argmin([err_base, err_alt]))
best_name = ["baseline", "alternative"][best_idx]
best_score = [err_base, err_alt][best_idx]
best_model = [base, alt][best_idx]

mlca_runtime.record_metric(primary_metric, best_score, "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")

Xs_val = get_scaled(X_val, best_model, pre, sel)
mlp = best_model.steps[-1][1] if isinstance(best_model, Pipeline) else best_model

def apply_activation(Z, activation):
    if activation == 'relu':
        return np.maximum(Z, 0)
    if activation == 'tanh':
        return np.tanh(Z)
    if activation == 'logistic':
        return 1 / (1 + np.exp(-Z))
    if activation == 'identity':
        return Z
    raise ValueError(f"Unknown activation: {activation}")

def get_embedding(mlp, X):
    A = X
    acts = []
    for i in range(len(mlp.coefs_) - 1):
        Z = A @ mlp.coefs_[i] + mlp.intercepts_[i]
        A = apply_activation(Z, mlp.activation)
        acts.append(A)
    sizes = [a.shape[1] for a in acts]
    return acts[int(np.argmin(sizes))]

emb = get_embedding(mlp, Xs_val)
emb_dim = emb.shape[1]

row_out = pd.DataFrame(emb, index=val.index)
row_out.columns = [f"emb_{i}" for i in range(emb_dim)]
row_out.index.name = "row_id"
mlca_runtime.save_row_outputs(row_out, "val_embeddings.parquet")

print(f"val recon err baseline={err_base:.4f} alt={err_alt:.4f}")
print(f"best={best_name} score={best_score:.4f} emb_dim={emb_dim}")
