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

feat_cols = [c for c in train.columns
             if c not in targets and c not in excluded
             and c != group_col and not is_keylike(train[c])]
X_val = val[feat_cols]

with open(inputs["05_preprocess_fit_preprocessing_M7FAZM1F/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_271P20QZ/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)
with open(inputs["07_model_train_baseline_DR0KT0NW/model.pkl"], "rb") as f:
    base = pickle.load(f)
with open(inputs["08_model_train_alternative_FRH197EZ/alternative_model.pkl"], "rb") as f:
    alt = pickle.load(f)

def last_est(m):
    return m.steps[-1][1] if isinstance(m, Pipeline) else m

def frontend(m):
    if isinstance(m, Pipeline) and len(m.steps) > 1:
        return Pipeline(m.steps[:-1])
    if isinstance(sel, Pipeline):
        return sel
    return Pipeline([("pre", pre), ("sel", sel)])

def reconstruct(m, X):
    est = last_est(m)
    if hasattr(est, "predict"):
        return m.predict(X)
    return m.inverse_transform(m.transform(X))

def recon_error(m, X):
    Xs = frontend(m).transform(X)
    Xr = reconstruct(m, X)
    return float(np.mean((Xs - Xr) ** 2))

def embedding(m, X):
    est = last_est(m)
    if not hasattr(est, "predict"):
        return m.transform(X)
    Xs = frontend(m).transform(X)
    A = Xs
    acts = []
    for i in range(len(est.coefs_) - 1):
        Z = A @ est.coefs_[i] + est.intercepts_[i]
        if est.activation == "relu":
            A = np.maximum(Z, 0)
        elif est.activation == "tanh":
            A = np.tanh(Z)
        elif est.activation == "logistic":
            A = 1 / (1 + np.exp(-Z))
        else:
            A = Z
        acts.append(A)
    sizes = [a.shape[1] for a in acts]
    return acts[int(np.argmin(sizes))]

err_base = recon_error(base, X_val)
err_alt = recon_error(alt, X_val)

results = pd.DataFrame({
    "model": ["baseline", "alternative"],
    "metric": [primary_metric, primary_metric],
    "value": [err_base, err_alt],
})
mlca_runtime.save_table(results, "validation_metrics.csv",
                        caption="Validation reconstruction error")

errs = [err_base, err_alt]
names = ["baseline", "alternative"]
best_idx = int(np.argmin(errs))
best_name = names[best_idx]
best_score = errs[best_idx]
best_model = [base, alt][best_idx]

mlca_runtime.record_metric(primary_metric, best_score, "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")

emb = embedding(best_model, X_val)
emb_dim = emb.shape[1]
row_out = pd.DataFrame(emb, index=val.index)
row_out.columns = [f"emb_{i}" for i in range(emb_dim)]
row_out.index.name = "row_id"
mlca_runtime.save_row_outputs(row_out, "val_embeddings.parquet")

print(f"val recon err baseline={err_base:.4f} alt={err_alt:.4f}")
print(f"best={best_name} score={best_score:.4f} emb_dim={emb_dim}")
