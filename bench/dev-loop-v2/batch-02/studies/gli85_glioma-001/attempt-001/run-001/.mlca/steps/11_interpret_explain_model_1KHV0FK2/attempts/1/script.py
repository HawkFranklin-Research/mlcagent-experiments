import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
prot = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in prot:
        raise KeyError(f"Missing protocol key {k!r}; available keys: {list(prot)}")
targets = list(prot["targets"])
target = targets[0]
excluded = set(prot["excluded_columns"])
group = prot.get("group_column")
task = prot["task"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
with open(inp["latest_model"], "rb") as f:
    model = pickle.load(f)

if hasattr(model, "feature_names_in_"):
    feat_cols = list(model.feature_names_in_)
else:
    drop = set(targets) | excluded | ({group} if group else set())
    feat_cols = [c for c in train.columns if c not in drop]
drop = set(targets) | excluded | ({group} if group else set())
feat_cols = [c for c in feat_cols if c not in drop]
X = val[feat_cols].copy()
y = val[target].copy()

def native_importance(model, cols):
    est = list(model.named_steps.values())[-1] if hasattr(model, "named_steps") and model.named_steps else model
    v = None
    if hasattr(est, "coef_"):
        c = np.asarray(est.coef_)
        v = np.abs(c[0]) if c.ndim == 2 and c.shape[0] == 1 else (np.abs(c).mean(axis=0) if c.ndim == 2 else np.abs(c))
    elif hasattr(est, "feature_importances_"):
        v = np.abs(np.asarray(est.feature_importances_))
    if v is None:
        return None
    if len(v) == len(cols):
        return pd.Series(v, index=cols)
    names = None
    try:
        if hasattr(model, "named_steps") and len(model.named_steps) > 1:
            from sklearn.pipeline import Pipeline
            pre = Pipeline(list(model.named_steps.items())[:-1])
            try:
                names = pre.get_feature_names_out(cols)
            except Exception:
                names = pre.get_feature_names_out()
        elif hasattr(model, "get_feature_names_out"):
            names = model.get_feature_names_out(cols)
    except Exception:
        names = None
    if names is not None and len(names) == len(v):
        raw = []
        for nm in names:
            s = str(nm)
            matches = [c for c in cols if str(c) in s]
            raw.append(max(matches, key=len) if matches else s)
        return pd.Series(v, index=raw).groupby(level=0).max()
    return None

def score_xy(Xdf):
    proba = model.predict_proba(Xdf)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    classes = np.asarray(getattr(model, "classes_", np.unique(y)))
    if proba.shape[1] == 2:
        return roc_auc_score(y, proba[:, 1], pos_label=classes[1] if len(classes) > 1 else 1)
    return roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)

base = score_xy(X)
native = native_importance(model, feat_cols)
ranked = native.sort_values(ascending=False).index.tolist() if native is not None else []
ranked = [c for c in ranked if c in feat_cols]

def perm_importance(feats, n_repeats):
    rng = np.random.default_rng(42)
    rows = []
    for col in feats:
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            deltas.append(score_xy(Xp) - base)
        rows.append({"feature": col, "importance_mean": float(np.mean(deltas)), "importance_sd": float(np.std(deltas))})
    return pd.DataFrame(rows)

if len(feat_cols) > 500 and not ranked:
    interp = pd.DataFrame([{"feature": "unavailable", "importance_mean": np.nan, "importance_sd": np.nan}])
    mlca_runtime.save_table(interp, "interpretation.csv", caption="Permutation importance unavailable; native ranking missing")
    mlca_runtime.record_fact_candidate("top_feature", "unavailable", source="interpretation.csv")
    print(f"train={len(train)} val={len(val)} features={len(feat_cols)}")
    print("No reliable native feature ranking; reported limitation")
elif len(feat_cols) > 500:
    n_repeats = 5
    interp = perm_importance(ranked[:50], n_repeats).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
    mlca_runtime.save_table(interp, "interpretation.csv", caption="Validation permutation importance, top 50 native-ranked features, 5 repeats")
    top_feature = str(interp.iloc[0]["feature"])
    mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
    print(f"train={len(train)} val={len(val)} features={len(feat_cols)}")
    print("method=permutation top50 native, repeats=5")
    print(f"top_feature={top_feature}")
else:
    n_repeats = 20
    interp = perm_importance(feat_cols, n_repeats).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
    mlca_runtime.save_table(interp, "interpretation.csv", caption="Validation permutation importance, all features, 20 repeats")
    top_feature = str(interp.iloc[0]["feature"])
    mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
    print(f"train={len(train)} val={len(val)} features={len(feat_cols)}")
    print("method=permutation all, repeats=20")
    print(f"top_feature={top_feature}")
