import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol)}")

targets = protocol["targets"]
task = protocol["task"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
print(f"train rows={len(train)}, val rows={len(val)}")

drop = set(targets) | set(excluded) | ({group} if group else set())

def is_keylike(s):
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == len(s)
    return False

feature_cols = [c for c in train.columns if c not in drop and not is_keylike(train[c])]
with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)
if hasattr(model, "feature_names_in_"):
    model_cols = list(model.feature_names_in_)
    if set(model_cols) != set(feature_cols):
        raise ValueError("Model input columns differ from frozen feature columns")
    feature_cols = model_cols
print(f"features={len(feature_cols)}")

target = targets[0]
val = val[val[target].notna()].copy()
y = val[target].to_numpy()
X = val[feature_cols]

def auc_score(est, X_, y_):
    p = est.predict_proba(X_)
    if p.shape[1] == 2:
        return roc_auc_score(y_, p[:, 1])
    return roc_auc_score(y_, p, multi_class="ovr", average="macro", labels=getattr(est, "classes_", None))

def own_importance(est, n):
    obj = est
    if hasattr(obj, "named_steps"):
        obj = list(obj.named_steps.values())[-1]
    for candidate in [est, obj]:
        if hasattr(candidate, "coef_"):
            c = np.asarray(candidate.coef_)
            if c.ndim > 1 and c.shape[-1] == n:
                return np.abs(c).mean(axis=0)
            if c.ndim == 1 and c.shape[0] == n:
                return np.abs(c)
        if hasattr(candidate, "feature_importances_"):
            f = np.asarray(candidate.feature_importances_)
            if f.shape[0] == n:
                return f
    return None

if len(feature_cols) <= 500:
    repeats = 20
    perm_features = feature_cols
    res = permutation_importance(model, X, y, scoring=auc_score, n_repeats=repeats, random_state=42, n_jobs=1)
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": res.importances_mean,
        "importance_sd": res.importances_std,
    })
else:
    own = own_importance(model, len(feature_cols))
    if own is None:
        raise RuntimeError("No reliable own importance ranking; refusing full-width permutation")
    top_idx = np.argsort(-own)[:50]
    perm_features = [feature_cols[i] for i in top_idx]
    repeats = 5
    base = auc_score(model, X, y)
    rng = np.random.default_rng(42)
    rows = []
    for f in perm_features:
        vals = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            vals.append(base - auc_score(model, Xp, y))
        rows.append((f, float(np.mean(vals)), float(np.std(vals))))
    imp = pd.DataFrame(rows, columns=["feature", "importance_mean", "importance_sd"])

imp = imp.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
mlca_runtime.save_table(imp, "interpretation.csv", caption="Top 15 validation permutation importances")
top_feature = str(imp.iloc[0]["feature"])
top_mean = float(imp.iloc[0]["importance_mean"])
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
mlca_runtime.record_metric("n_features_permuted", float(len(perm_features)), "val")
mlca_runtime.record_metric("n_repeats", float(repeats), "val")
mlca_runtime.record_metric("top_importance_mean", top_mean, "val")
print(f"method=permutation_importance, features={len(perm_features)}, repeats={repeats}")
print(f"top_feature={top_feature}, mean={top_mean:.4g}")
