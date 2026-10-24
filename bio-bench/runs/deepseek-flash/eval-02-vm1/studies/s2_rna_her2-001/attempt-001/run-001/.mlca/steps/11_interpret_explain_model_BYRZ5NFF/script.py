import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = list(protocol["targets"])
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

def raw_columns(model, train):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    drop = set(targets) | {"sample_id"}
    group = protocol.get("group_column")
    if group:
        drop.add(group)
    return [c for c in train.columns if c not in drop]

raw_cols = raw_columns(model, train)
X_val = val[raw_cols].copy()
y_val = val[target]

def classes_of(model):
    if hasattr(model, "classes_"):
        return list(model.classes_)
    if hasattr(model, "named_steps"):
        final = list(model.named_steps.values())[-1]
        return list(getattr(final, "classes_", []))
    return []

def score_model(m, X, y):
    proba = m.predict_proba(X)
    classes = classes_of(m)
    if proba.shape[1] == 2:
        y_bin = (y == classes[1]).astype(int) if len(classes) == 2 else y
        return float(roc_auc_score(y_bin, proba[:, 1]))
    return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))

def raw_importance(model, raw_cols):
    est = model
    names = None
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.items())
        est = steps[-1][1]
        try:
            names = list(model[:-1].get_feature_names_out())
        except Exception:
            names = None
    if hasattr(est, "coef_"):
        imp = np.abs(np.asarray(est.coef_))
        if imp.ndim > 1:
            imp = imp.mean(axis=0)
    elif hasattr(est, "feature_importances_"):
        imp = np.abs(np.asarray(est.feature_importances_))
    else:
        return None
    imp = imp.ravel()
    if len(imp) == len(raw_cols):
        return raw_cols, imp
    if hasattr(model, "named_steps"):
        for name, step in list(model.named_steps.items())[:-1]:
            if hasattr(step, "get_support"):
                try:
                    support = step.get_support()
                except Exception:
                    continue
                if len(support) == len(raw_cols) and int(support.sum()) == len(imp):
                    return [c for c, s in zip(raw_cols, support) if s], imp
    if names is not None and len(names) == len(imp):
        mapping = {}
        raw_set = set(raw_cols)
        for n, v in zip(names, imp):
            k = n if n in raw_set else (n.split("__")[-1] if "__" in n else n)
            if k not in raw_set:
                cand = [r for r in raw_cols if k.endswith("__" + r) or k.endswith(r)]
                k = cand[0] if cand else None
            if k in raw_set:
                mapping[k] = max(mapping.get(k, 0.0), float(v))
        if mapping:
            keys = list(mapping)
            return keys, np.array([mapping[k] for k in keys])
    return None

def perm_importance(model, X, y, features, n_repeats, seed):
    rng = np.random.default_rng(seed)
    Xp = X.copy()
    base = score_model(model, Xp, y)
    rows = []
    for feat in features:
        original = Xp[feat].to_numpy(copy=True)
        scores = []
        for _ in range(n_repeats):
            shuffled = original.copy()
            rng.shuffle(shuffled)
            Xp[feat] = shuffled
            scores.append(score_model(model, Xp, y))
            Xp[feat] = original
        drops = base - np.asarray(scores)
        sd = float(np.std(drops, ddof=1)) if n_repeats > 1 else 0.0
        rows.append((feat, float(np.mean(drops)), sd))
    return pd.DataFrame(rows, columns=["feature", "importance_mean", "importance_sd"])

n_features = len(raw_cols)
if n_features > 500:
    ranked = raw_importance(model, raw_cols)
    if ranked is None:
        df_all = pd.DataFrame(columns=["feature", "importance_mean", "importance_sd"])
        method = "missing reliable coefficient/importance ranking; no full-width permutation"
    else:
        feats, imps = ranked
        order = np.argsort(imps)[::-1][:50]
        top_feats = [feats[i] for i in order]
        df_all = perm_importance(model, X_val, y_val, top_feats, 5, 42)
        method = "own importance top-50 permutation, n_repeats=5"
else:
    df_all = perm_importance(model, X_val, y_val, raw_cols, 20, 42)
    method = f"permutation all {n_features} features, n_repeats=20"

df_top = df_all.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
if not df_top.empty:
    df_top["importance_mean"] = df_top["importance_mean"].round(4)
    df_top["importance_sd"] = df_top["importance_sd"].round(4)

mlca_runtime.save_table(
    df_top,
    "interpretation.csv",
    caption=f"Feature importance; method={method}; n_features={n_features}",
)
if not df_top.empty:
    mlca_runtime.record_fact_candidate(
        "top_feature", str(df_top.loc[0, "feature"]), source="interpretation.csv"
    )

print(f"train={len(train)} val={len(val)}")
print(f"n_features={n_features}")
print(method)
if not df_top.empty:
    print(f"top_feature={df_top.loc[0, 'feature']}")
