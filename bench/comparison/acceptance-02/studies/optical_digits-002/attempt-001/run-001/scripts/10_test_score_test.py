import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
task = proto["task"]
targets = proto.get("targets", []) or []
excluded = proto.get("excluded_columns", []) or []
group_col = proto.get("group_column")
metric_name = task["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
test = pd.read_parquet(inp["test"])

skip = set(targets) | set(excluded) | ({group_col} if group_col else set())
feature_cols = []
for c in train.columns:
    if c in skip: continue
    s = train[c]
    if s.dtype == object: continue
    if s.dtype.kind in "iu" and s.nunique() > 1 and s.nunique() == len(s): continue
    feature_cols.append(c)
print(f"features: {len(feature_cols)}")

pre = pickle.loads(inp["05_preprocess_fit_preprocessing_W49DB36F/preprocessor.pkl"].read_bytes())
sel = pickle.loads(inp["06_features_retain_features_XVA92YG5/selector_all.pkl"].read_bytes())

def preps_raw(o):
    fn = getattr(o, "feature_names_in_", None)
    return fn is not None and all(c in set(fn) for c in feature_cols)

frontend = sel if preps_raw(sel) else Pipeline([("pre", pre), ("sel", sel)])

def final_est(o):
    return o.steps[-1][1] if isinstance(o, Pipeline) else o

m1 = final_est(pickle.loads(inp["07_model_train_baseline_EGH0EDWF/model.pkl"].read_bytes()))
m2 = final_est(pickle.loads(inp["08_model_train_alternative_X1KB5C0A/alternative_model.pkl"].read_bytes()))

def act(a, name):
    if name == "relu": return np.maximum(a, 0.0)
    if name == "tanh": return np.tanh(a)
    if name == "logistic": return 1.0 / (1.0 + np.exp(-a))
    return a

def bottleneck(mlp, Z):
    A = np.asarray(Z, dtype=float)
    stop = (len(mlp.coefs_) + 1) // 2
    for i in range(stop):
        A = act(A @ mlp.coefs_[i] + mlp.intercepts_[i], mlp.activation)
    return A

def evaluate(model, Z):
    Zf = np.asarray(Z, dtype=float)
    if hasattr(model, "inverse_transform"):
        emb = model.transform(Zf)
        recon = model.inverse_transform(emb)
    else:
        recon = model.predict(Zf)
        emb = bottleneck(model, Zf)
    recon = np.asarray(recon, dtype=float)
    emb = np.asarray(emb, dtype=float)
    if emb.ndim == 1: emb = emb.reshape(-1, 1)
    if recon.ndim == 1: recon = recon.reshape(-1, 1)
    return float(np.mean((recon - Zf) ** 2)), emb

Z_val = frontend.transform(val[feature_cols])
Z_train = frontend.transform(train[feature_cols])
Z_test = frontend.transform(test[feature_cols])

val_scores = {"baseline": evaluate(m1, Z_val)[0], "alternative": evaluate(m2, Z_val)[0]}
best_name = min(val_scores, key=val_scores.get)
best = {"baseline": m1, "alternative": m2}[best_name]
print("val:", {k: round(v, 4) for k, v in val_scores.items()})
print("best:", best_name)

mse_test, emb_test = evaluate(best, Z_test)
_, emb_train = evaluate(best, Z_train)

mlca_runtime.record_metric(metric_name, mse_test, "test")
mlca_runtime.record_fact_candidate("test_score", mse_test, source="test_metrics.csv")
mlca_runtime.save_table(
    pd.DataFrame([{"model": best_name, "metric": metric_name, "value": mse_test}]),
    "test_metrics.csv",
)

d = emb_test.shape[1]
cols = [f"embedding_{i}" for i in range(d)]
tr = pd.DataFrame(emb_train, columns=cols)
tr.insert(0, "row_id", train.index.to_numpy())
tr.insert(1, "partition", "train")
te = pd.DataFrame(emb_test, columns=cols)
te.insert(0, "row_id", test.index.to_numpy())
te.insert(1, "partition", "test")
mlca_runtime.save_row_outputs(pd.concat([tr, te], ignore_index=True), "embeddings.parquet")
print(f"test {metric_name}: {mse_test:.4f}  emb_dim: {d}")
