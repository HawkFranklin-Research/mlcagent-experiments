import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
try:
    targets = protocol["targets"]; excluded = protocol["excluded_columns"]; task = protocol["task"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available top-level keys: {list(protocol.keys())}")
group_col = protocol.get("group_column")
target = targets[0]
primary_metric = task.get("primary_metric", "roc_auc_ovr")
seed = 42

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
print(f"train rows={len(train)} val rows={len(val)}")

def key_like(s):
    if pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.notna().all() and s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.notna().all() and s.nunique(dropna=True) == len(s) and (s.max() - s.min() + 1 == len(s))
    return False

target_set, excluded_set = set(targets), set(excluded)
feature_cols = [c for c in train.columns
                if c not in target_set and c not in excluded_set and c != group_col and not key_like(train[c])]

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)
model_cols = list(getattr(model, "feature_names_in_", feature_cols))
if set(model_cols) & (target_set | excluded_set):
    raise ValueError("Model input schema contains target or excluded columns")
X_val = val[model_cols].copy()
y_val = val[target]

def scorer(est, X, y):
    p = est.predict_proba(X)
    if p.shape[1] == 2:
        return roc_auc_score(y, p[:, 1])
    return roc_auc_score(y, p, multi_class="ovr", average="macro")

def extract_ranking(est, cols):
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_)
        if len(imp) == len(cols):
            return pd.Series(imp, index=cols).sort_values(ascending=False)
    if hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        if len(imp) == len(cols):
            return pd.Series(imp, index=cols).sort_values(ascending=False)
    if isinstance(est, Pipeline):
        final = est.named_steps[list(est.named_steps)[-1]]
        imp = None
        if hasattr(final, "coef_"):
            coef = np.asarray(final.coef_)
            imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        elif hasattr(final, "feature_importances_"):
            imp = np.asarray(final.feature_importances_)
        if imp is None:
            return None
        try:
            names = [str(n) for n in est[:-1].get_feature_names_out(cols)]
        except Exception:
            return None
        raw = set(cols); mapped = []
        for n in names:
            cand = n if n in raw else n.split("__")[-1]
            if cand not in raw:
                matches = [c for c in cols if n.endswith(c) or n.endswith("__" + c)]
                cand = matches[0] if matches else None
            mapped.append(cand)
        if any(c is None for c in mapped):
            return None
        s = pd.Series(0.0, index=cols)
        for c, v in zip(mapped, imp):
            s[c] += float(v)
        return s.sort_values(ascending=False)
    return None

n_features = len(model_cols)
if n_features <= 500:
    method = "validation_permutation_importance"
    n_repeats = 20
    imp = permutation_importance(model, X_val, y_val, scoring=scorer,
                                 n_repeats=n_repeats, random_state=seed, n_jobs=1)
    imp_df = pd.DataFrame({"feature": model_cols,
                           "importance_mean": imp.importances_mean,
                           "importance_sd": imp.importances_std})
    imp_df = imp_df.sort_values("importance_mean", ascending=False).head(15).round(4)
else:
    ranking = extract_ranking(model, model_cols)
    if ranking is None:
        method, n_repeats = "unavailable_no_reliable_ranking", 0
        imp_df = pd.DataFrame(columns=["feature", "importance_mean", "importance_sd"])
    else:
        method = "top_50_coefficient_ranked_permutation"
        n_repeats = 5
        top_feats = list(ranking.index[:50])
        rng = np.random.default_rng(seed)
        baseline = scorer(model, X_val, y_val)
        rows = []
        for feat in top_feats:
            vals = []
            for _ in range(n_repeats):
                Xp = X_val.copy(deep=False)
                Xp[feat] = rng.permutation(X_val[feat].to_numpy())
                vals.append(baseline - scorer(model, Xp, y_val))
            rows.append((feat, float(np.mean(vals)), float(np.std(vals))))
        imp_df = pd.DataFrame(rows, columns=["feature", "importance_mean", "importance_sd"])
        imp_df = imp_df.sort_values("importance_mean", ascending=False).head(15).round(4)

caption = f"method={method}; features_permuted={min(n_features, 50) if n_features > 500 else n_features}; n_repeats={n_repeats}; metric={primary_metric}"
mlca_runtime.save_table(imp_df, "interpretation.csv", caption=caption)
if len(imp_df):
    mlca_runtime.record_fact_candidate("top_feature", str(imp_df.iloc[0]["feature"]), source="interpretation.csv")
print(f"method={method} features={n_features} repeats={n_repeats} top15={len(imp_df)}")
