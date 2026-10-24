import json, pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
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

def get_classes(m):
    if hasattr(m, "classes_"):
        return list(m.classes_)
    if hasattr(m, "named_steps"):
        last = list(m.named_steps.values())[-1]
        if hasattr(last, "classes_"):
            return list(last.classes_)
    return None

cls = get_classes(model)
if cls is not None and len(cls) >= 2:
    mapping = {c: i for i, c in enumerate(cls)}
    y_enc = y_raw.map(mapping).to_numpy()
    if np.isnan(y_enc).any():
        y_enc = (y_raw == y_raw.iloc[0]).astype(int).to_numpy()
else:
    y_enc = (y_raw == y_raw.iloc[0]).astype(int).to_numpy()

if task["type"] == "binary":
    def scorer(est, X, y_true):
        return roc_auc_score(y_true, est.predict_proba(X)[:, 1])
else:
    def scorer(est, X, y_true):
        return roc_auc_score(y_true, est.predict_proba(X), average="macro")

def own_importance(m, n):
    if hasattr(m, "coef_"):
        c = np.asarray(m.coef_)
        v = np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c)
        if len(v) == n:
            return v
    if hasattr(m, "feature_importances_"):
        v = np.asarray(m.feature_importances_)
        if len(v) == n:
            return v
    if hasattr(m, "named_steps"):
        for s in m.named_steps.values():
            if hasattr(s, "coef_"):
                c = np.asarray(s.coef_)
                v = np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c)
                if len(v) == n:
                    return v
            if hasattr(s, "feature_importances_"):
                v = np.asarray(s.feature_importances_)
                if len(v) == n:
                    return v
    return None

def manual_perm(model, Xv, yv, feats, n_reps, seed=42):
    base = scorer(model, Xv, yv)
    rng = np.random.RandomState(seed)
    ms, ss = [], []
    for feat in feats:
        sc = []
        for _ in range(n_reps):
            Xp = Xv.copy()
            Xp[feat] = Xp[feat].values[rng.permutation(len(Xp))]
            sc.append(scorer(model, Xp, yv))
        ms.append(base - np.mean(sc))
        ss.append(np.std(sc))
    return np.array(ms), np.array(ss)

n_f = len(mcols)
if n_f <= 500:
    method, n_rep, n_perm = "permutation_importance", 20, n_f
    r = permutation_importance(model, X_val, y_enc, scoring=scorer,
                               n_repeats=n_rep, random_state=42, n_jobs=1)
    names, means, sds = mcols, r.importances_mean, r.importances_std
else:
    coefs = own_importance(model, n_f)
    if coefs is None:
        method, n_rep, n_perm = "unavailable", 0, 0
        names, means, sds = [], np.array([]), np.array([])
        mlca_runtime.record_metric("ranking_unavailable", 1.0, "val")
    else:
        top = [mcols[i] for i in np.argsort(coefs)[-50:][::-1]]
        method, n_rep, n_perm = "top50_permutation", 5, len(top)
        m_, s_ = manual_perm(model, X_val, y_enc, top, n_rep, seed=42)
        names, means, sds = top, m_, s_

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
