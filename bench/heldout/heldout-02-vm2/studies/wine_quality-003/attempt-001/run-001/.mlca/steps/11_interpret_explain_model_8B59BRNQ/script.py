import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, make_scorer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
proto = json.loads(inputs["_protocol"].read_text())

targets = proto["targets"]
excluded = proto["excluded_columns"]
group = proto.get("group_column")
task = proto["task"]
metric = task["primary_metric"]
print(f"train {train.shape} val {val.shape} metric {metric}")

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
feature_cols = [c for c in feature_cols if not (
    (train[c].dtype == object or str(train[c].dtype).startswith("str"))
    and train[c].nunique() == len(train))]

with open(inputs["07_model_train_logistic_5W0X2YFF/model.pkl"], "rb") as f:
    model = pickle.load(f)

in_names = list(getattr(model, "feature_names_in_", feature_cols))
X_val = val[in_names].copy()
y_val = val[targets[0]].values

if metric == "roc_auc_ovr":
    def scorer(est, X, y):
        p = est.predict_proba(X)
        return roc_auc_score(y, p, multi_class="ovr", average="macro")
else:
    scorer = make_scorer(roc_auc_score, needs_proba=True)

n_feat = len(in_names)
if n_feat > 500:
    imp = getattr(model, "feature_importances_", None)
    if imp is None and hasattr(model, "coef_"):
        imp = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
    if imp is None:
        imp = np.zeros(n_feat)
    order = np.argsort(-np.asarray(imp))[:min(50, n_feat)]
    perm_cols = [in_names[i] for i in order]
    n_repeats = 5
    method = "top50_native_importance_permutation"
else:
    perm_cols = list(in_names)
    n_repeats = 20
    method = "permutation_importance"

X_perm = X_val.copy()
ref_scores = np.array([scorer(model, X_perm, y_val) for _ in range(n_repeats)])
imp_mean = {}
imp_sd = {}
for col in perm_cols:
    scores = []
    orig = X_perm[col].values.copy()
    for _ in range(n_repeats):
        X_perm[col] = np.random.permutation(orig)
        scores.append(scorer(model, X_perm, y_val))
    X_perm[col] = orig
    d = ref_scores - np.array(scores)
    imp_mean[col] = float(np.mean(d))
    imp_sd[col] = float(np.std(d))

res = pd.DataFrame({"feature": list(imp_mean.keys()),
                    "importance_mean": list(imp_mean.values()),
                    "importance_sd": list(imp_sd.values())})
res = res.sort_values("importance_mean", ascending=False).head(50).reset_index(drop=True)
for c in ["importance_mean", "importance_sd"]:
    res[c] = res[c].round(6)
mlca_runtime.save_table(res, "interpretation.csv", caption=f"{method}, n_repeats={n_repeats}")

top = res.iloc[0]["feature"]
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
print(f"method={method} features={len(perm_cols)} repeats={n_repeats} top={top}")
