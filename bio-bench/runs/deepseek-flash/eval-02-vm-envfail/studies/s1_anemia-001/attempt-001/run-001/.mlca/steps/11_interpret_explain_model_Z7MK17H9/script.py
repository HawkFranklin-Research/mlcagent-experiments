import json, pickle, numpy as np, pandas as pd, mlca_runtime
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
targets = protocol["targets"]
excluded = set(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
task = protocol["task"]

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    if (s.dtype == object or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == s.notna().sum():
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s) and s.is_monotonic_increasing:
        continue
    feature_cols.append(c)

with open(paths["latest_model"], "rb") as f:
    model = pickle.load(f)

model_cols = list(getattr(model, "feature_names_in_", feature_cols))
missing = [c for c in model_cols if c not in val.columns]
if missing:
    raise ValueError(f"Model input columns missing from validation: {missing}")

X = val[model_cols].copy()
y = val[targets[0]]

def final_est(m):
    if hasattr(m, "named_steps"):
        return list(m.named_steps.values())[-1]
    return m

classes = getattr(model, "classes_", None)
if classes is None:
    classes = getattr(final_est(model), "classes_", None)

def score_model(Xdf):
    proba = model.predict_proba(Xdf)
    if task.get("n_classes", 2) == 2:
        yb = (y == classes[1]).astype(int) if classes is not None and len(classes) == 2 else y
        return roc_auc_score(yb, proba[:, 1])
    return roc_auc_score(y, proba, multi_class="ovr", average="macro")

if len(model_cols) > 500:
    est = final_est(model)
    imp = None
    names = getattr(est, "feature_names_in_", None)
    if hasattr(est, "coef_"):
        a = np.asarray(est.coef_)
        a = np.abs(a).mean(axis=0) if a.ndim > 1 else np.abs(a)
        if a.size == len(model_cols) and (not hasattr(model, "named_steps") or (names is not None and list(names) == model_cols)):
            imp = a
    if imp is None and hasattr(est, "feature_importances_"):
        a = np.asarray(est.feature_importances_).ravel()
        if a.size == len(model_cols) and (not hasattr(model, "named_steps") or (names is not None and list(names) == model_cols)):
            imp = a
    if imp is not None:
        candidates = [model_cols[i] for i in np.argsort(-imp)[:50]]
        n_repeats = 5
        method = "top50_permutation"
    else:
        candidates, n_repeats, method = [], 0, "ranking_unavailable"
else:
    candidates, n_repeats, method = model_cols, 20, "full_permutation"

rows = []
if candidates:
    base = score_model(X)
    for j, c in enumerate(candidates):
        vals = []
        for r in range(n_repeats):
            Xp = X.copy()
            rng = np.random.default_rng(42 + 1009 * r + j)
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            vals.append(base - score_model(Xp))
        rows.append({
            "feature": c,
            "importance_mean": float(np.mean(vals)) if vals else np.nan,
            "importance_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        })

out = pd.DataFrame(rows, columns=["feature", "importance_mean", "importance_sd"]).sort_values(
    "importance_mean", ascending=False
).head(15).reset_index(drop=True)
for c in ["importance_mean", "importance_sd"]:
    out[c] = out[c].map(lambda v: float(f"{v:.4g}") if pd.notna(v) else np.nan)

mlca_runtime.save_table(out, "interpretation.csv", caption="Permutation importance of frozen model features")
top_feature = str(out.iloc[0]["feature"]) if len(out) else "unavailable"
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
mlca_runtime.record_metric("interpretation_n_features", float(len(candidates)), "val")
mlca_runtime.record_metric("interpretation_n_repeats", float(n_repeats), "val")

print(f"val_rows={len(val)} model_features={len(model_cols)}")
print(f"method={method} permuted_features={len(candidates)} repeats={n_repeats}")
print(f"top_feature={top_feature}")
