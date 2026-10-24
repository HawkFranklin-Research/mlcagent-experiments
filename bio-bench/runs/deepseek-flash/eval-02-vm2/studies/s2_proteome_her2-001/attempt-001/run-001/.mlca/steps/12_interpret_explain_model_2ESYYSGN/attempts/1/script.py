import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
task = protocol["task"]
metric = task["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
print("train", train.shape, "val", val.shape)

y = val[targets[0]] if targets[0] in val.columns else None
target_col = targets[0]

# Load frozen model (most complete: latest_model preferred)
def try_load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

model = try_load(inp["latest_model"])
feat_names = list(getattr(model, "feature_names_in_", []))
if not feat_names:
    # fall back to reconstructing from train
    pass

# Feature columns per RL-0007
excl = set(excluded) | set(targets)
if "sample_id" in train.columns:
    excl.add("sample_id")
feat_cols = [c for c in train.columns if c not in excl]
# keep only those the model accepts
if feat_names:
    feat_cols = [c for c in feat_names if c in val.columns]

X_val = val[feat_cols].copy()

def scorer(est, X, y_true):
    if metric == "roc_auc" and task.get("n_classes", 2) == 2:
        p = est.predict_proba(X)[:, 1]
        return roc_auc_score(y_true, p)
    else:
        p = est.predict_proba(X)
        return roc_auc_score(y_true, p, multi_class="ovr", average="macro")

n_features = len(feat_cols)
if n_features > 500:
    # use own importances / coefficients
    est = model
    if hasattr(est, "named_steps"):
        last = list(est.named_steps.values())[-1]
    else:
        last = est
    imp = None
    if hasattr(last, "feature_importances_"):
        imp = np.asarray(last.feature_importances_)
    elif hasattr(last, "coef_"):
        coef = np.asarray(last.coef_)
        imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    if imp is not None and len(imp) == len(feat_cols):
        order = np.argsort(imp)[::-1][:50]
        top_cols = [feat_cols[i] for i in order]
        n_rep = 5
        res = permutation_importance(model, X_val[top_cols], y,
                                     scoring=scorer, n_repeats=n_rep,
                                     random_state=42, n_jobs=1)
        mean_imp = res.importances_mean
        sd_imp = res.importances_std
        scores = dict(zip(top_cols, zip(mean_imp, sd_imp)))
        method = f"permutation_top50_{n_rep}rep"
    else:
        # cannot rank reliably
        ranking = pd.DataFrame(columns=["feature","importance_mean","importance_sd"])
        mlca_runtime.save_table(ranking, "interpretation.csv", caption="unavailable")
        mlca_runtime.record_fact_candidate("top_feature", "unavailable",
                                            source="interpretation.csv")
        mlca_runtime.save_json({"method": "unavailable",
                                "n_features": n_features},
                               "interpretation_meta.json")
        raise SystemExit("reliable ranking unavailable")
else:
    n_rep = 20
    res = permutation_importance(model, X_val, y, scoring=scorer,
                                 n_repeats=n_rep, random_state=42, n_jobs=1)
    scores = dict(zip(feat_cols, zip(res.importances_mean, res.importances_std)))
    method = f"permutation_all_{n_rep}rep"

rows = [{"feature": f, "importance_mean": float(m), "importance_sd": float(s)}
        for f, (m, s) in scores.items()]
ranking = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).head(15)
ranking["importance_mean"] = ranking["importance_mean"].round(4)
ranking["importance_sd"] = ranking["importance_sd"].round(4)

mlca_runtime.save_table(ranking, "interpretation.csv", caption="Top permutation importances")
mlca_runtime.save_json({"method": method, "n_features": n_features,
                         "n_repeats": n_rep if n_features <= 500 else 5},
                        "interpretation_meta.json")
if len(ranking):
    mlca_runtime.record_fact_candidate("top_feature", ranking.iloc[0]["feature"],
                                        source="interpretation.csv")
print("method", method, "n_features", n_features, "top", ranking.iloc[0]["feature"] if len(ranking) else None)
