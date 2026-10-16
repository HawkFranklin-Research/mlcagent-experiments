import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
targets = proto["targets"]; excluded = proto["excluded_columns"]
group = proto.get("group_column"); primary = proto["task"]["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
test = pd.read_parquet(inp["test"])

drop = set(targets) | set(excluded)
if group: drop.add(group)
def keylike(s):
    if s.dtype == object or pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False
feat_cols = [c for c in train.columns if c not in drop and not keylike(train[c])]

def load(p):
    with open(p, "rb") as f: return pickle.load(f)
pre = load(inp["05_preprocess_fit_preprocessing_E2NYE0HC/preprocessor.pkl"])
sel = load(inp["06_features_retain_features_8M4G64VT/selector_all.pkl"])
m_base = load(inp["07_model_train_baseline_JYP4DTKE/model.pkl"])
m_alt = load(inp["08_model_train_alternative_29T9VJ6F/alternative_model.pkl"])

def build_frontend():
    if isinstance(sel, Pipeline) and any(s is pre for s in sel.named_steps.values()):
        return sel
    if isinstance(pre, Pipeline) and any(s is sel for s in pre.named_steps.values()):
        return pre
    return Pipeline([("pre", pre), ("sel", sel)])
frontend = build_frontend()

def get_est(m):
    return m.steps[-1][1] if isinstance(m, Pipeline) else m

def dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

def recon_err(est, Xf):
    if isinstance(est, PCA):
        R = est.inverse_transform(est.transform(Xf))
    elif isinstance(est, MLPRegressor):
        R = est.predict(Xf)
    elif hasattr(est, "inverse_transform") and hasattr(est, "transform"):
        R = est.inverse_transform(est.transform(Xf))
    else:
        R = est.predict(Xf)
    return float(np.mean((dense(Xf) - dense(R)) ** 2))

Xf_train = frontend.transform(train[feat_cols])
Xf_val = frontend.transform(val[feat_cols])
Xf_test = frontend.transform(test[feat_cols])

cands = {"baseline": get_est(m_base), "alternative": get_est(m_alt)}
val_scores = {k: recon_err(e, Xf_val) for k, e in cands.items()}
winner_name = min(val_scores, key=val_scores.get)
winner = cands[winner_name]
test_score = recon_err(winner, Xf_test)

mlca_runtime.save_table(
    pd.DataFrame([{"model": winner_name, "metric": primary, "value": test_score}]),
    "test_metrics.csv", caption="Test reconstruction error")
mlca_runtime.record_metric(primary, test_score, "test")
mlca_runtime.record_fact_candidate("test_score", test_score, source="test_metrics.csv")

def mlp_embed(mlp, Xf):
    A = dense(Xf).astype(float)
    widths = [c.shape[1] for c in mlp.coefs_[:-1]]
    mid = int(np.argmin(widths))
    for i in range(mid + 1):
        A = A @ mlp.coefs_[i] + mlp.intercepts_[i]
        A = np.maximum(0, A)
    return A

if isinstance(winner, PCA):
    E_tr, E_te = winner.transform(Xf_train), winner.transform(Xf_test)
else:
    E_tr, E_te = mlp_embed(winner, Xf_train), mlp_embed(winner, Xf_test)
E_tr, E_te = dense(E_tr), dense(E_te)

cols = [f"embedding_{i}" for i in range(E_tr.shape[1])]
dtr = pd.DataFrame(E_tr, columns=cols); dtr["row_id"] = train.index; dtr["partition"] = "train"
dte = pd.DataFrame(E_te, columns=cols); dte["row_id"] = test.index; dte["partition"] = "test"
emb = pd.concat([dtr, dte], ignore_index=True)[["row_id", "partition"] + cols]
mlca_runtime.save_row_outputs(emb, "embeddings.parquet")

print(f"features={len(feat_cols)} val={val_scores}")
print(f"winner={winner_name} test_{primary}={test_score:.4f}")
print(f"embeddings={emb.shape}")
