import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

protocol = json.loads(paths["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
task = protocol["task"]
metric = task["primary_metric"]

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
print("rows", len(train), len(val))

with open(paths["latest_model"], "rb") as f:
    model = pickle.load(f)

feat_cols = [c for c in train.columns
             if c not in targets and c not in excluded and c != group_col]
target = targets[0]

# frozen model input schema
if hasattr(model, "feature_names_in_"):
    model_cols = [c for c in model.feature_names_in_ if c in feat_cols]
else:
    model_cols = feat_cols
X_val = val[model_cols].copy()
y_val = val[target].values

n_feats = len(model_cols)
rng = np.random.RandomState(42)

def score_fn(est, X, y):
    p = est.predict_proba(X)
    if task["n_classes"] == 2:
        return roc_auc_score(y, p[:, 1])
    return roc_auc_score(y, p, multi_class="ovr", average="macro")

if n_feats <= 500:
    n_rep = 20
    candidates = model_cols
    method = "permutation_importance_full"
else:
    # rank by own importances; permute top 50 with n_repeats<=5
    imp = None
    inner = model
    if hasattr(inner, "named_steps"):
        for name in reversed(list(inner.named_steps)):
            est = inner.named_steps[name]
            if hasattr(est, "feature_importances_"):
                imp = np.asarray(est.feature_importances_); break
            if hasattr(est, "coef_"):
                c = np.asarray(est.coef_)
                imp = np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c); break
    if imp is not None and len(imp) == n_feats:
        top_idx = np.argsort(imp)[::-1][:50]
        candidates = [model_cols[i] for i in top_idx]
    else:
        candidates = []
    n_rep = 5
    method = "permutation_importance_top50"

if len(candidates) == 0:
    print("No reliable ranking")
    pd.DataFrame(columns=["feature", "importance_mean", "importance_sd"]).to_csv(
        out / "interpretation.csv", index=False
    )
    mlca_runtime.save_table(
        pd.DataFrame({"note": ["no reliable ranking available"]}),
        "interpretation.csv"
    )
    mlca_runtime.record_fact_candidate("top_feature", "unavailable",
                                       source="interpretation.csv")
else:
    res = permutation_importance(
        model, X_val, y_val, scoring=score_fn, n_repeats=n_rep,
        random_state=42, n_jobs=1
    )
    imp_df = pd.DataFrame({
        "feature": candidates,
        "importance_mean": res.importances_mean,
        "importance_sd": res.importances_std,
    }).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
    imp_df = imp_df.round(4)
    mlca_runtime.save_table(imp_df, "interpretation.csv",
                            caption=f"{method}, n_rep={n_rep}, n_feats={n_feats}")
    top = str(imp_df.iloc[0]["feature"]) if len(imp_df) else "unavailable"
    mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
    print("top", top, "n", n_feats, "reps", n_rep)
