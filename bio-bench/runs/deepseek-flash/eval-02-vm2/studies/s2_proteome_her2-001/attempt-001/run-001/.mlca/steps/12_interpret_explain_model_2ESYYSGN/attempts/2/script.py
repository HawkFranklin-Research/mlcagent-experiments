import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
task = protocol["task"]
target = protocol["targets"][0]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
y = val[target].values

with open(inp["latest_model"], "rb") as f:
    model = pickle.load(f)

feat_cols = list(getattr(model, "feature_names_in_", []))
if not feat_cols:
    excl = set(protocol.get("excluded_columns", [])) | set(protocol["targets"]) | {protocol.get("group_column")} | {"sample_id"}
    feat_cols = [c for c in train.columns if c not in excl]
X_val = val[feat_cols].copy()
n_features = len(feat_cols)

def scorer(est, X, y_true):
    p = est.predict_proba(X)
    if task.get("n_classes", 2) == 2:
        return roc_auc_score(y_true, p[:, 1])
    return roc_auc_score(y_true, p, multi_class="ovr", average="macro")

if n_features <= 500:
    n_rep = 20
    res = permutation_importance(model, X_val, y, scoring=scorer, n_repeats=n_rep, random_state=42, n_jobs=1)
    ranking = pd.DataFrame({"feature": feat_cols, "importance_mean": res.importances_mean, "importance_sd": res.importances_std})
    ranking = ranking.sort_values("importance_mean", ascending=False).head(15)
    method = f"permutation_all_{n_rep}rep"
else:
    final = model
    if hasattr(model, "named_steps"):
        final = list(model.named_steps.values())[-1]
    imp = None
    if hasattr(final, "coef_"):
        c = np.asarray(final.coef_)
        imp = np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c)
    elif hasattr(final, "feature_importances_"):
        imp = np.asarray(final.feature_importances_)
    aligned = None
    if imp is not None:
        if len(imp) == n_features:
            aligned = imp
        elif hasattr(model, "named_steps") and "selector" in model.named_steps:
            sel = model.named_steps["selector"]
            support = sel.get_support(indices=True) if hasattr(sel, "get_support") else None
            pre = model.named_steps.get("preprocessor")
            if support is not None and pre is not None and hasattr(pre, "get_feature_names_out"):
                names = pre.get_feature_names_out()
                aligned = np.zeros(n_features)
                for i, c in zip(support, imp):
                    if i < len(names) and names[i] in feat_cols:
                        aligned[feat_cols.index(names[i])] = c
    if aligned is not None:
        order = np.argsort(aligned)[::-1][:15]
        ranking = pd.DataFrame({"feature": [feat_cols[i] for i in order],
                                "importance_mean": aligned[order],
                                "importance_sd": 0.0})
        method = "coefficients"
    else:
        ranking = pd.DataFrame(columns=["feature", "importance_mean", "importance_sd"])
        method = "unavailable"

for col in ["importance_mean", "importance_sd"]:
    if col in ranking:
        ranking[col] = ranking[col].round(4)

mlca_runtime.save_table(ranking, "interpretation.csv", caption="Top feature importances")
mlca_runtime.save_json({"method": method, "n_features": n_features,
                        "n_repeats": 20 if n_features <= 500 else 0},
                       "interpretation_meta.json")
if len(ranking):
    mlca_runtime.record_fact_candidate("top_feature", str(ranking.iloc[0]["feature"]), source="interpretation.csv")
else:
    mlca_runtime.record_fact_candidate("top_feature", "unavailable", source="interpretation.csv")
print("method", method, "n_features", n_features, "top", ranking.iloc[0]["feature"] if len(ranking) else None)
