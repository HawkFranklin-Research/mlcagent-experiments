import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network._base import ACTIVATIONS

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

proto = json.loads(inp["_protocol"].read_text())
targets = proto["targets"]
excluded = proto["excluded_columns"]
group = proto.get("group_column")
primary_metric = proto["task"]["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
test = pd.read_parquet(inp["test"])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feat_cols = [c for c in train.columns if c not in drop]

def is_key_like(s):
    if s.dtype == object or pd.api.types.is_integer_dtype(s):
        return s.nunique() == len(s)
    return False

feat_cols = [c for c in feat_cols if not is_key_like(train[c])]

with open(inp["05_preprocess_fit_preprocessing_RDQ93D4J/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_retain_features_0YD36CMR/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)
with open(inp["07_model_train_baseline_EE9J97EQ/model.pkl"], "rb") as f:
    m_base = pickle.load(f)
with open(inp["08_model_train_alternative_N9ZERH70/alternative_model.pkl"], "rb") as f:
    m_alt = pickle.load(f)

def build_frontend(pre, sel):
    if isinstance(pre, Pipeline) and any(s is sel for s in pre.named_steps.values()):
        return pre
    if isinstance(sel, Pipeline) and any(s is pre for s in sel.named_steps.values()):
        return sel
    return Pipeline([("pre", pre), ("sel", sel)])

frontend = build_frontend(pre, sel)

def input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for s in obj.named_steps.values():
            if hasattr(s, "feature_names_in_"):
                return list(s.feature_names_in_)
    return None

cols = input_cols(frontend) or feat_cols
Xtr = frontend.transform(train[cols])
Xva = frontend.transform(val[cols])
Xte = frontend.transform(test[cols])

def get_est(model):
    if isinstance(model, Pipeline):
        return model.named_steps[list(model.named_steps)[-1]]
    return model

def model_has_frontend(model):
    if not isinstance(model, Pipeline):
        return False
    steps = list(model.named_steps.values())
    return any(s is pre for s in steps) or any(s is sel for s in steps)

def get_frontend_X(model, raw_df, our_Xf):
    if model_has_frontend(model):
        mfe = Pipeline(list(model.named_steps.items())[:-1])
        mcols = input_cols(mfe) or cols
        return mfe.transform(raw_df[mcols])
    return our_Xf

def reconstruct(model, Xf):
    est = get_est(model)
    if isinstance(est, PCA):
        return est.inverse_transform(est.transform(Xf))
    return est.predict(Xf)

def embed(model, Xf):
    est = get_est(model)
    if isinstance(est, PCA):
        return est.transform(Xf)
    if isinstance(est, MLPRegressor):
        act = ACTIVATIONS[est.activation]
        return act(Xf @ est.coefs_[0] + est.intercepts_[0])
    return est.transform(Xf)

candidates = {"baseline": m_base, "alternative": m_alt}
val_scores = {}
for name, m in candidates.items():
    Xv = get_frontend_X(m, val, Xva)
    R = reconstruct(m, Xv)
    val_scores[name] = float(np.mean((Xv - R) ** 2))

best = min(val_scores, key=val_scores.get)
best_model = candidates[best]

Xt = get_frontend_X(best_model, test, Xte)
Rt = reconstruct(best_model, Xt)
test_mse = float(np.mean((Xt - Rt) ** 2))

metrics_df = pd.DataFrame([{"model": best, "metric": primary_metric, "value": round(test_mse, 6)}])
mlca_runtime.save_table(metrics_df, "test_metrics.csv", "Test reconstruction error")
mlca_runtime.record_metric(primary_metric, test_mse, "test")
mlca_runtime.record_fact_candidate("test_score", test_mse, source="test_metrics.csv")

emb_tr = embed(best_model, get_frontend_X(best_model, train, Xtr))
emb_te = embed(best_model, get_frontend_X(best_model, test, Xte))

def make_emb_df(emb, idx, part):
    df = pd.DataFrame(emb, columns=[f"embedding_{i}" for i in range(emb.shape[1])])
    df.insert(0, "row_id", idx)
    df.insert(1, "partition", part)
    return df

emb_all = pd.concat([make_emb_df(emb_tr, train.index, "train"),
                     make_emb_df(emb_te, test.index, "test")], ignore_index=True)
mlca_runtime.save_row_outputs(emb_all, "embeddings.parquet")

print(f"val scores: {val_scores}")
print(f"best: {best}, test {primary_metric}={test_mse:.6f}")
print(f"embeddings shape: {emb_all.shape}")
