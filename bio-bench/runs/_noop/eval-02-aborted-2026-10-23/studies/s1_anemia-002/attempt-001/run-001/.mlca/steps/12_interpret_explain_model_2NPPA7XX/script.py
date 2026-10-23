import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
seed = 42
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
primary_metric = protocol["task"]["primary_metric"]
target = targets[0]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
with open(inp["latest_model"], "rb") as f:
    model = pickle.load(f)

feat_cols = list(getattr(model, "feature_names_in_", []))
if not feat_cols:
    forbidden = set(targets) | set(excluded)
    if group:
        forbidden.add(group)
    feat_cols = [c for c in train.columns if c not in forbidden]
for c in forbidden if "forbidden" in locals() else set(targets) | set(excluded):
    if c in feat_cols:
        raise ValueError(f"forbidden column in frozen model input schema: {c}")
missing = [c for c in feat_cols if c not in val.columns]
if missing:
    raise KeyError(f"missing frozen input columns in val: {missing}")

X = val[feat_cols]
y = val[target]
classes = getattr(model, "classes_", None)
if classes is None:
    classes = np.unique(y)
classes = np.asarray(classes)

def score_fn(est, X_, y_):
    proba = est.predict_proba(X_)
    y_arr = np.asarray(y_)
    if len(classes) == 2:
        return roc_auc_score((y_arr == classes[1]).astype(int), proba[:, 1])
    return roc_auc_score(y_arr, proba, multi_class="ovr", average="macro", labels=classes)

def subset_perm_importance(est, X_, y_, cols, n_repeats):
    rng = np.random.default_rng(seed)
    base = score_fn(est, X_, y_)
    rows = []
    for col in cols:
        vals = []
        for _ in range(n_repeats):
            Xp = X_.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            vals.append(score_fn(est, Xp, y_))
        rows.append((col, base - float(np.mean(vals)), float(np.std(vals))))
    return pd.DataFrame(rows, columns=["feature", "importance_mean", "importance_sd"])

n_features = len(feat_cols)
if n_features <= 500:
    pi = permutation_importance(
        model, X, y, scoring=score_fn, n_repeats=20,
        random_state=seed, n_jobs=1
    )
    interp = pd.DataFrame({
        "feature": feat_cols,
        "importance_mean": pi.importances_mean,
        "importance_sd": pi.importances_std,
    })
    method = "validation_permutation_importance"
    budget = 20
else:
    base_est = model
    if hasattr(model, "named_steps"):
        base_est = list(model.named_steps.values())[-1]
    imp = getattr(base_est, "feature_importances_", None)
    if imp is None:
        coef = getattr(base_est, "coef_", None)
        if coef is not None:
            imp = np.abs(coef).mean(axis=0) if np.ndim(coef) > 1 else np.abs(coef)
    if imp is not None and len(imp) == n_features:
        top_idx = np.argsort(imp)[::-1][:50]
        top_cols = [feat_cols[i] for i in top_idx]
        budget = 5
        interp = subset_perm_importance(model, X, y, top_cols, budget)
        method = "subset_permutation_top50"
    else:
        interp = pd.DataFrame(columns=["feature", "importance_mean", "importance_sd"])
        method = "no_reliable_ranking"
        budget = 0

interp = interp.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
for c in ["importance_mean", "importance_sd"]:
    interp[c] = interp[c].map(lambda v: float(f"{v:.4g}"))
mlca_runtime.save_table(interp, "interpretation.csv", caption="Validation permutation importance")
if len(interp):
    mlca_runtime.record_fact_candidate("top_feature", interp.loc[0, "feature"], source="interpretation.csv")

print(f"train_rows={len(train)} val_rows={len(val)}")
print(f"n_features={n_features} method={method} n_repeats={budget}")
if len(interp):
    print(f"top_feature={interp.loc[0, 'feature']} importance_mean={interp.loc[0, 'importance_mean']}")
else:
    print("limitation=no reliable frozen feature ranking available")
