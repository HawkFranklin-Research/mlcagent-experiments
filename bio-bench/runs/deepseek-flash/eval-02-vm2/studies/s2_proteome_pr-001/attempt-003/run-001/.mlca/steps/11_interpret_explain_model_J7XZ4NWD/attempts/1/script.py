import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
targets = protocol["targets"]
target = targets[0]
excluded_val = protocol["excluded_columns"]
if isinstance(excluded_val, list):
    excluded = set(excluded_val)
elif isinstance(excluded_val, dict):
    for key in ("columns", "entries", "values"):
        if isinstance(excluded_val.get(key), list):
            excluded = set(excluded_val[key])
            break
    else:
        raise ValueError(f"excluded_columns dict keys: {list(excluded_val.keys())}")
else:
    raise ValueError("excluded_columns must be a list or dict")
group = protocol.get("group_column")
task = protocol["task"]
n_classes = int(task.get("n_classes", 2))

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

def keylike(s):
    if pd.api.types.is_integer_dtype(s) or s.dtype == object or str(s.dtype).startswith("string"):
        n = s.nunique(dropna=False)
        return n > 1 and n == len(s)
    return False

excluded_all = set(targets) | excluded | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in excluded_all and not keylike(train[c])]

with open(paths["latest_model"], "rb") as f:
    model = pickle.load(f)
model_cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else feature_cols
bad = set(model_cols) & excluded_all
if bad:
    raise ValueError(f"model schema contains target/excluded/group: {sorted(bad)}")
missing = set(model_cols) - set(feature_cols)
if missing:
    raise ValueError(f"model columns outside frozen feature definition: {sorted(missing)}")

X_val = val[model_cols].copy()
y_val = val[target]
classes = getattr(model, "classes_", np.unique(y_val))
mapping = {c: i for i, c in enumerate(classes)}
y_enc = np.array([mapping[v] for v in y_val])

def score_fn(est, X, y):
    proba = est.predict_proba(X)
    if proba.shape[1] == 2:
        return roc_auc_score(y, proba[:, 1])
    return roc_auc_score(y, proba, multi_class="ovr", average="macro")

def extract_importance(est, cols):
    def arr_from(e):
        if hasattr(e, "feature_importances_"):
            return np.asarray(e.feature_importances_, dtype=float)
        if hasattr(e, "coef_"):
            c = np.asarray(e.coef_, dtype=float)
            return np.abs(c).mean(axis=0) if c.ndim == 2 and c.shape[0] > 1 else np.abs(c).ravel()
        return None
    a = arr_from(est)
    if a is not None and len(a) == len(cols):
        return pd.Series(a, index=cols)
    if hasattr(est, "named_steps"):
        steps = list(est.named_steps.values())
        a = arr_from(steps[-1])
        if a is None:
            return None
        for step in steps[:-1]:
            if hasattr(step, "get_support") and hasattr(step, "feature_names_in_"):
                names = np.asarray(step.feature_names_in_)
                sup = step.get_support()
                if len(sup) == len(names) and int(sup.sum()) == len(a):
                    sel = names[sup]
                    if len(set(sel) & set(cols)) >= max(1, int(0.5 * len(sel))):
                        return pd.Series(a, index=sel).reindex(cols).fillna(0.0)
        if len(a) == len(cols):
            return pd.Series(a, index=cols)
    return None

n_input = len(model_cols)
baseline = None
if n_input <= 500:
    perm_features = model_cols
    n_repeats = 20
    method = "validation_permutation_importance"
    pi = permutation_importance(
        model, X_val, y_enc, scoring=score_fn, n_repeats=n_repeats,
        random_state=42, n_jobs=1,
    )
    result = pd.DataFrame({
        "feature": model_cols,
        "importance_mean": pi.importances_mean,
        "importance_sd": pi.importances_std,
    })
    baseline = float(score_fn(model, X_val, y_enc))
    result = result.sort_values("importance_mean", ascending=False).head(15)
else:
    rank = extract_importance(model, model_cols)
    if rank is None:
        result = pd.DataFrame([{
            "feature": "unavailable",
            "importance_mean": np.nan,
            "importance_sd": np.nan,
        }])
        perm_features = []
        n_repeats = 0
        method = "limitation_no_reliable_ranking"
        mlca_runtime.record_metric("importance_ranking_available", 0.0, split="val")
    else:
        perm_features = rank.sort_values(ascending=False).head(50).index.tolist()
        n_repeats = 5
        method = "coefficient_ranked_permutation_top50"
        baseline = float(score_fn(model, X_val, y_enc))
        rng = np.random.RandomState(42)
        rows = []
        for feat in perm_features:
            scores = []
            for _ in range(n_repeats):
                Xp = X_val.copy()
                Xp[feat] = rng.permutation(Xp[feat].to_numpy())
                scores.append(score_fn(model, Xp, y_enc))
            rows.append({
                "feature": feat,
                "importance_mean": baseline - float(np.mean(scores)),
                "importance_sd": float(np.std(scores, ddof=1)) if n_repeats > 1 else 0.0,
            })
        result = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).head(15)
        mlca_runtime.record_metric("importance_ranking_available", 1.0, split="val")

result = result[["feature", "importance_mean", "importance_sd"]].copy()
result["importance_mean"] = result["importance_mean"].round(6)
result["importance_sd"] = result["importance_sd"].round(6)
mlca_runtime.save_table(result, "interpretation.csv", caption=f"Feature importance: {method}")
top_feature = str(result.iloc[0]["feature"]) if len(result) else "unavailable"
mlca_runtime.record_metric("n_features_permuted", float(len(perm_features)), split="val")
mlca_runtime.record_metric("n_repeats", float(n_repeats), split="val")
mlca_runtime.record_metric("input_feature_count", float(n_input), split="val")
if baseline is not None and not np.isnan(baseline):
    mlca_runtime.record_metric("baseline_validation_auc", float(baseline), split="val")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"train {train.shape}, val {val.shape}; input features {n_input}")
print(f"method {method}; permuted {len(perm_features)} features x {n_repeats} repeats")
print(f"top feature: {top_feature}")
