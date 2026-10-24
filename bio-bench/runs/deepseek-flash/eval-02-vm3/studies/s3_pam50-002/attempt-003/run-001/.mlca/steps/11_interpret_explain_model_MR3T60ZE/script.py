import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, roc_auc_score

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
prot = json.loads(paths["_protocol"].read_text())
needed = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in needed if k not in prot]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available {list(prot.keys())}")

targets = prot["targets"]
target_col = targets[0] if isinstance(targets, list) else targets
excluded = set(prot["excluded_columns"])
group = prot["group_column"]
primary_metric = prot["task"]["primary_metric"]

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
with open(paths["latest_model"], "rb") as f:
    model = pickle.load(f)

protected = set(targets) | excluded | ({group} if group else set())

def key_like(col, s):
    if col in protected:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s) and (s.is_monotonic_increasing or "id" in col.lower())
    return False

feature_cols = [c for c in train.columns if c not in protected and not key_like(c, train[c])]

model_cols = getattr(model, "feature_names_in_", None)
model_cols = list(model_cols) if model_cols is not None else feature_cols
missing_cols = [c for c in model_cols if c not in val.columns]
if missing_cols:
    raise KeyError(f"Frozen model input columns missing from val: {missing_cols}")

X_val = val.loc[:, model_cols].copy()
y_val = val[target_col]
scorer = make_scorer(roc_auc_score, response_method="predict_proba", multi_class="ovr", average="macro")

if len(model_cols) <= 500:
    result = permutation_importance(
        model, X_val, y_val, scoring=scorer, n_repeats=20, random_state=42, n_jobs=1
    )
    imp_df = pd.DataFrame({
        "feature": model_cols,
        "importance_mean": result.importances_mean,
        "importance_sd": result.importances_std,
    })
    method = f"validation permutation importance; features={len(model_cols)}; n_repeats=20"
else:
    final = list(model.named_steps.values())[-1] if hasattr(model, "named_steps") and model.named_steps else model
    imp = None
    if hasattr(final, "coef_"):
        coef = np.asarray(final.coef_)
        imp = np.abs(coef) if coef.ndim == 1 else np.mean(np.abs(coef), axis=0)
    elif hasattr(final, "feature_importances_"):
        imp = np.asarray(final.feature_importances_)

    if imp is not None and len(imp) == len(model_cols):
        order = np.argsort(imp)[::-1][:50]
        top_cols = [model_cols[i] for i in order]
        rng = np.random.default_rng(42)
        base = scorer(model, X_val, y_val)
        means, sds = [], []
        for col in top_cols:
            vals = X_val[col].to_numpy(copy=True)
            scores = []
            for _ in range(5):
                Xp = X_val.copy()
                Xp[col] = rng.permutation(vals)
                scores.append(scorer(model, Xp, y_val))
            means.append(base - float(np.mean(scores)))
            sds.append(float(np.std(scores)))
        imp_df = pd.DataFrame({
            "feature": top_cols,
            "importance_mean": means,
            "importance_sd": sds,
        })
        method = "model coefficient/feature_importance ranking; top50 permuted; n_repeats=5"
    else:
        imp_df = pd.DataFrame({
            "feature": model_cols[:15],
            "importance_mean": np.nan,
            "importance_sd": np.nan,
        })
        method = "reliable full-width ranking unavailable; no permutation performed"

top15 = imp_df.sort_values("importance_mean", ascending=False, na_position="last").head(15).copy()
for c in ["importance_mean", "importance_sd"]:
    top15[c] = top15[c].map(lambda x: float(f"{x:.4g}") if pd.notna(x) else x)

mlca_runtime.save_table(top15, "interpretation.csv", caption=method)
top_feature = str(top15.iloc[0]["feature"]) if len(top15) and pd.notna(top15.iloc[0]["importance_mean"]) else "unavailable"
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"train={len(train)} val={len(val)} model_features={len(model_cols)}")
print(method)
print(f"top_feature={top_feature}")
