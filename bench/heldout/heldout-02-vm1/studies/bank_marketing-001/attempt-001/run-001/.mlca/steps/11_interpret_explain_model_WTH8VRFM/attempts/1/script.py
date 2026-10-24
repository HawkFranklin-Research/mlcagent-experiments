import json, pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
task = protocol["task"]
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feat_cols = [c for c in train.columns if c not in drop]
print(f"train={len(train)} val={len(val)} feats={len(feat_cols)}")

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

def input_cols(m):
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    if hasattr(m, "named_steps"):
        for s in m.named_steps.values():
            if hasattr(s, "feature_names_in_"):
                return list(s.feature_names_in_)
    return feat_cols

mcols = input_cols(model)
X_val = val[mcols].copy()
y_raw = val[targets[0]]
le = LabelEncoder()
y_enc = le.fit_transform(y_raw)

if task["type"] == "binary":
    def scorer(est, X, y_true):
        return roc_auc_score(y_true, est.predict_proba(X)[:, 1])
else:
    def scorer(est, X, y_true):
        return roc_auc_score(y_true, est.predict_proba(X), average="macro")

n_f = len(mcols)
if n_f <= 500:
    method, n_rep, n_perm = "permutation_importance", 20, n_f
    r = permutation_importance(model, X_val, y_enc, scoring=scorer,
                               n_repeats=n_rep, random_state=42, n_jobs=1)
    names, means, sds = mcols, r.importances_mean, r.importances_std
else:
    coefs = None
    if hasattr(model, "coef_"):
        c = np.asarray(model.coef_)
        coefs = np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c)
    elif hasattr(model, "feature_importances_"):
        coefs = np.asarray(model.feature_importances_)
    if coefs is None or len(coefs) != n_f:
        method, n_rep, n_perm = "unavailable", 0, 0
        names, means, sds = [], np.array([]), np.array([])
        mlca_runtime.record_metric("ranking_unavailable", 1.0, "val")
    else:
        top = [mcols[i] for i in np.argsort(coefs)[-50:][::-1]]
        method, n_rep, n_perm = "top50_permutation", 5, len(top)
        base = scorer(model, X_val, y_enc)
        rng = np.random.RandomState(42)
        m_, s_ = [], []
        for feat in top:
            sc = []
            for _ in range(n_rep):
                Xp = X_val.copy()
                Xp[feat] = Xp[feat].values[rng.permutation(len(Xp))]
                sc.append(scorer(model, Xp, y_enc))
            m_.append(base - np.mean(sc))
            s_.append(np.std(sc))
        names, means, sds = top, np.array(m_), np.array(s_)

df = pd.DataFrame({"feature": names, "importance_mean": means, "importance_sd": sds})
df = df.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
df["importance_mean"] = df["importance_mean"].round(6)
df["importance_sd"] = df["importance_sd"].round(6)
mlca_runtime.save_table(df, "interpretation.csv")
mlca_runtime.record_metric("n_features_permuted", float(n_perm), "val")
mlca_runtime.record_metric("n_repeats", float(n_rep), "val")
if len(df):
    mlca_runtime.record_fact_candidate("top_feature", str(df.iloc[0]["feature"]),
                                       source="interpretation.csv")
    print(f"method={method} n_perm={n_perm} n_rep={n_rep} top={df.iloc[0]['feature']}")
