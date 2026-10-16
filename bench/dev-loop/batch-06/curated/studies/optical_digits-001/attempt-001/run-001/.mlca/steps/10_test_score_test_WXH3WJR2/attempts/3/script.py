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
targets = proto["targets"]; excluded = proto["excluded_columns"]
group = proto.get("group_column"); primary_metric = proto["task"]["primary_metric"]

train = pd.read_parquet(inp["train"]); val = pd.read_parquet(inp["val"]); test = pd.read_parquet(inp["test"])
drop = set(targets) | set(excluded) | ({group} if group else set())
feat_cols = [c for c in train.columns if c not in drop]
feat_cols = [c for c in feat_cols if not ((train[c].dtype == object or pd.api.types.is_integer_dtype(train[c])) and train[c].nunique() == len(train))]

def load(k):
    with open(inp[k], "rb") as f: return pickle.load(f)
pre = load("05_preprocess_fit_preprocessing_RDQ93D4J/preprocessor.pkl")
sel = load("06_features_retain_features_0YD36CMR/selector_all.pkl")
m_base = load("07_model_train_baseline_EE9J97EQ/model.pkl")
m_alt = load("08_model_train_alternative_N9ZERH70/alternative_model.pkl")

def build_frontend(pre, sel):
    if isinstance(pre, Pipeline) and any(s is sel for s in pre.named_steps.values()): return pre
    if isinstance(sel, Pipeline) and any(s is pre for s in sel.named_steps.values()): return sel
    return Pipeline([("pre", pre), ("sel", sel)])
frontend = build_frontend(pre, sel)

def input_cols(obj):
    if hasattr(obj, "feature_names_in_"): return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for s in obj.named_steps.values():
            if hasattr(s, "feature_names_in_"): return list(s.feature_names_in_)
    return None
cols = input_cols(frontend) or feat_cols

def get_est(m): return m.named_steps[list(m.named_steps)[-1]] if isinstance(m, Pipeline) else m
def has_frontend(m):
    return isinstance(m, Pipeline) and any(s is pre or s is sel for s in m.named_steps.values())
def frontend_X(m, raw, Xf):
    if has_frontend(m):
        mfe = Pipeline(list(m.named_steps.items())[:-1])
        return mfe.transform(raw[input_cols(mfe) or cols])
    return Xf
def reconstruct(m, Xf):
    e = get_est(m)
    return e.inverse_transform(e.transform(Xf)) if isinstance(e, PCA) else e.predict(Xf)
def embed(m, Xf):
    e = get_est(m)
    if isinstance(e, PCA): return e.transform(Xf)
    if isinstance(e, MLPRegressor): return ACTIVATIONS[e.activation](Xf @ e.coefs_[0] + e.intercepts_[0])
    return e.transform(Xf)

Xtr = frontend.transform(train[cols]); Xva = frontend.transform(val[cols]); Xte = frontend.transform(test[cols])
cands = {"baseline": m_base, "alternative": m_alt}
val_scores = {}
for n, m in cands.items():
    Xv = frontend_X(m, val, Xva); R = reconstruct(m, Xv)
    val_scores[n] = float(np.mean((Xv - R) ** 2))
best = min(val_scores, key=val_scores.get); bm = cands[best]
Xt = frontend_X(bm, test, Xte); Rt = reconstruct(bm, Xt)
test_mse = float(np.mean((Xt - Rt) ** 2))

mlca_runtime.save_table(pd.DataFrame([{"model": best, "metric": primary_metric, "value": round(test_mse, 6)}]),
                        "test_metrics.csv", "Test reconstruction error")
mlca_runtime.record_metric(primary_metric, test_mse, "test")
mlca_runtime.record_fact_candidate("test_score", test_mse, source="test_metrics.csv")

def emb_df(emb, idx, part):
    df = pd.DataFrame(emb, columns=[f"embedding_{i}" for i in range(emb.shape[1])])
    df.insert(0, "row_id", idx); df.insert(1, "partition", part); return df
emb_all = pd.concat([emb_df(embed(bm, frontend_X(bm, train, Xtr)), train.index, "train"),
                     emb_df(embed(bm, Xt), test.index, "test")], ignore_index=True)
mlca_runtime.save_row_outputs(emb_all, "embeddings.parquet")

print(f"val: {val_scores}")
print(f"best={best} test {primary_metric}={test_mse:.6f}")
print(f"embeddings {emb_all.shape}")
