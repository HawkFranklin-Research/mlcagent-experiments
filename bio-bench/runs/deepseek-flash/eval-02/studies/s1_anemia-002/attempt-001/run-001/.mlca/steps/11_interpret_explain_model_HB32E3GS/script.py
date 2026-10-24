import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
prot = json.loads(inputs["_protocol"].read_text())
targets = prot["targets"]
excluded = prot["excluded_columns"]
metric = prot["task"]["primary_metric"]
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
target = targets[0]
y_raw = val[target]

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

def input_columns(model, frame):
    cols = None
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
    elif hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                cols = list(step.feature_names_in_)
                break
    if cols is None:
        group = prot.get("group_column")
        cols = [c for c in frame.columns if c not in targets and c not in excluded and c != group]
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"Frozen model input columns missing from validation: {missing}")
    if any(c in targets or c in excluded for c in cols):
        raise ValueError("Frozen model schema includes target or excluded column")
    return cols

cols = input_columns(model, val)
Xval = val[cols]

classes = getattr(model, "classes_", None)
if classes is None and hasattr(model, "named_steps"):
    classes = list(model.named_steps.values())[-1].classes_
if classes is None:
    uniq = np.sort(pd.unique(y_raw.dropna()))
    if len(uniq) != 2:
        raise ValueError("Binary target expected")
    y = (y_raw == uniq[1]).astype(int).to_numpy()
else:
    if len(classes) != 2:
        raise ValueError("Binary target expected")
    y = (y_raw == classes[1]).astype(int).to_numpy()

proba = model.predict_proba(Xval)
baseline = roc_auc_score(y, proba[:, 1])
n_features = len(cols)
rng = np.random.default_rng(42)

if n_features <= 500:
    n_repeats = 20
    perm_cols = cols
    method = "validation_permutation_all_features"
else:
    est = model
    if hasattr(model, "named_steps"):
        est = list(model.named_steps.values())[-1]
    imp = None
    if hasattr(est, "coef_"):
        imp = np.ravel(est.coef_)
    elif hasattr(est, "feature_importances_"):
        imp = np.ravel(est.feature_importances_)
    if imp is not None and len(imp) == n_features:
        order = np.argsort(-np.abs(imp))[:50]
        perm_cols = [cols[i] for i in order]
        n_repeats = 5
        method = "validation_permutation_top50_model_importance"
    else:
        perm_cols = cols[:50]
        n_repeats = 5
        method = "limited_permutation_no_reliable_model_ranking"
        print("No reliable model importance ranking; permuting first 50 features only")

rows = []
for c in perm_cols:
    drops = []
    for _ in range(n_repeats):
        Xp = Xval.copy()
        Xp[c] = rng.permutation(Xp[c].to_numpy())
        p = model.predict_proba(Xp)[:, 1]
        drops.append(baseline - roc_auc_score(y, p))
    rows.append((c, float(np.mean(drops)),
                 float(np.std(drops, ddof=1)) if n_repeats > 1 else 0.0))

res = pd.DataFrame(rows, columns=["feature", "importance_mean", "importance_sd"])
res = res.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
for c in ["importance_mean", "importance_sd"]:
    res[c] = res[c].apply(lambda v: float(f"{v:.4g}"))

caption = (f"Validation permutation importance; metric={metric}; method={method}; "
           f"n_features={n_features}; permuted_features={len(perm_cols)}; repeats={n_repeats}; "
           f"baseline_auc={baseline:.4g}")
mlca_runtime.save_table(res, "interpretation.csv", caption=caption)
mlca_runtime.record_metric("baseline_val_auc", float(baseline), "val")
mlca_runtime.record_metric("n_permuted_features", float(len(perm_cols)), "val")
mlca_runtime.record_metric("permutation_repeats", float(n_repeats), "val")
top = res.iloc[0]["feature"] if len(res) else None
if top is not None:
    mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
print(f"val rows={len(val)} features={n_features} permuted={len(perm_cols)} repeats={n_repeats} baseline_auc={baseline:.4f}")
print(f"top_feature={top}")
