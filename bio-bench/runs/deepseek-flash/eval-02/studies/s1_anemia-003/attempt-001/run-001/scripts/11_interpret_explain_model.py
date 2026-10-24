import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
req = ["targets", "excluded_columns", "task"]
missing = [k for k in req if k not in protocol]
if missing:
    raise ValueError(f"Missing protocol keys {missing}; available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)
metric = protocol["task"].get("primary_metric", "roc_auc")
if not targets:
    raise ValueError("No target declared in protocol['targets']")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

def is_key_like(s):
    if s.nunique(dropna=True) != len(s) or len(s) == 0:
        return False
    return pd.api.types.is_string_dtype(s.dtype) or pd.api.types.is_integer_dtype(s.dtype)

if hasattr(model, "feature_names_in_"):
    cols = list(model.feature_names_in_)
else:
    drop = set(targets + excluded + ([group] if group else []))
    cols = [c for c in train.columns if c not in drop and not is_key_like(train[c])]
missing_cols = [c for c in cols if c not in val.columns]
if missing_cols:
    raise ValueError(f"Validation is missing frozen model input columns: {missing_cols}")

X_val = val[cols].copy()
y_val = val[targets[0]].copy()
classes = list(model.classes_) if hasattr(model, "classes_") else sorted(y_val.dropna().unique())
if len(classes) == 2:
    pos_label = classes[1]
    def scorer(est, X, y):
        p = est.predict_proba(X)[:, 1]
        return roc_auc_score((pd.Series(y).to_numpy() == pos_label).astype(int), p)
else:
    def scorer(est, X, y):
        return roc_auc_score(y, est.predict_proba(X), multi_class="ovr", average="macro")

n_features = len(cols)
if n_features <= 500:
    n_repeats = 20
    imp = permutation_importance(
        model, X_val, y_val, scoring=scorer, n_repeats=n_repeats,
        random_state=42, n_jobs=1
    )
    used_features, means, sds = cols, imp.importances_mean, imp.importances_std
else:
    n_repeats = 5
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "coef_"):
        co = np.abs(np.asarray(est.coef_))
        raw_imp = co.mean(axis=0) if co.ndim > 1 else co
    elif hasattr(est, "feature_importances_"):
        raw_imp = np.asarray(est.feature_importances_)
    else:
        raw_imp = None
    if raw_imp is None or len(raw_imp) != n_features:
        raise ValueError("No reliable feature ranking for >500 features; refusing full-width permutation.")
    top_idx = np.argsort(raw_imp)[::-1][:50]
    used_features = [cols[i] for i in top_idx]
    rng = np.random.default_rng(42)
    means, sds = [], []
    for feat in used_features:
        base = X_val[feat].to_numpy()
        scores = []
        for _ in range(n_repeats):
            Xp = X_val.copy()
            Xp[feat] = rng.permutation(base)
            scores.append(scorer(model, Xp, y_val))
        means.append(float(np.mean(scores)))
        sds.append(float(np.std(scores)))
    means, sds = np.array(means), np.array(sds)

out = pd.DataFrame({
    "feature": used_features,
    "importance_mean": means,
    "importance_sd": sds,
}).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
out[["importance_mean", "importance_sd"]] = out[["importance_mean", "importance_sd"]].round(6)

mlca_runtime.save_table(out, "interpretation.csv", caption="Top 15 validation permutation importances")
mlca_runtime.record_fact_candidate("top_feature", str(out.iloc[0]["feature"]), source="interpretation.csv")
mlca_runtime.record_metric("interpretation_n_features", float(n_features), "val")
mlca_runtime.record_metric("interpretation_n_repeats", float(n_repeats), "val")

print(f"features={n_features} repeats={n_repeats} val_rows={len(X_val)}")
print(f"top_feature={out.iloc[0]['feature']} importance={out.iloc[0]['importance_mean']:.4g}")
