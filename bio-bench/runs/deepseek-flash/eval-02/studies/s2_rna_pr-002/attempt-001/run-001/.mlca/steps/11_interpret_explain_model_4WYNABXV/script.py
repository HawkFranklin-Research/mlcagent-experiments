import json, pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets, task = protocol["targets"], protocol["task"]
excluded = set(protocol["excluded_columns"])
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
y = val[targets[0]].to_numpy()

model_key = "latest_model" if "latest_model" in inputs else "07_model_train_logistic_RTDQBRJK/model.pkl"
with open(inputs[model_key], "rb") as f:
    model = pickle.load(f)

def final_est(m):
    return m.steps[-1][1] if isinstance(m, Pipeline) else m

est = final_est(model)
classes = getattr(est, "classes_", None)

if hasattr(model, "feature_names_in_"):
    feats = list(model.feature_names_in_)
else:
    drop = set(targets) | excluded
    if "group_column" in protocol:
        drop.add(protocol["group_column"])
    feats = [c for c in train.columns if c not in drop]
feats = [c for c in feats if c in val.columns and c not in excluded and c not in targets]
X = val[feats].copy()

def score_est(m, Xs, ys):
    proba = m.predict_proba(Xs)
    if proba.ndim == 1:
        proba = np.column_stack([1 - proba, proba])
    if task.get("n_classes", 2) == 2:
        if classes is not None and len(classes) == 2:
            yb = (np.asarray(ys) == classes[1]).astype(int)
        else:
            yb = (np.asarray(ys) == np.unique(ys)[1]).astype(int)
        return roc_auc_score(yb, proba[:, 1])
    return roc_auc_score(ys, proba, multi_class="ovr", average="macro", labels=classes)

def map_imp(vals, names, raw):
    agg = defaultdict(float)
    raw_set = set(raw)
    for nm, v in zip(names, vals):
        s = str(nm)
        if s not in raw_set:
            s2 = s.split("__")[-1]
            if s2 in raw_set:
                s = s2
            else:
                continue
        agg[s] += abs(float(v))
    if not agg:
        return None
    return pd.DataFrame({"feature": list(agg), "importance_mean": list(agg.values())}).sort_values(
        "importance_mean", ascending=False
    )

def model_importances(m, raw):
    e = final_est(m)
    if hasattr(e, "coef_"):
        v = np.asarray(e.coef_)
        v = np.abs(v).mean(axis=0) if v.ndim > 1 and v.shape[0] > 1 else np.abs(v).ravel()
    elif hasattr(e, "feature_importances_"):
        v = np.abs(np.asarray(e.feature_importances_))
    else:
        return None
    names = None
    if isinstance(m, Pipeline) and len(m.steps) > 1:
        try:
            names = list(m[:-1].get_feature_names_out(input_features=raw))
        except Exception:
            names = None
    if names is None or len(names) != len(v):
        if len(v) == len(raw):
            names = raw
        else:
            return None
    return map_imp(v, names, raw)

def selector_ranking(raw):
    try:
        with open(inputs["06_features_select_features_8X366KXZ/selector.pkl"], "rb") as f:
            sel = pickle.load(f)
        with open(inputs["05_preprocess_fit_preprocessing_WM2D925J/preprocessor.pkl"], "rb") as f:
            pre = pickle.load(f)
    except Exception:
        return None
    if hasattr(sel, "scores_"):
        scores = np.abs(np.asarray(sel.scores_))
    elif hasattr(sel, "feature_importances_"):
        scores = np.abs(np.asarray(sel.feature_importances_))
    else:
        return None
    try:
        names = list(pre.get_feature_names_out(input_features=raw))
    except Exception:
        names = raw if len(scores) == len(raw) else None
    if names is None or len(names) != len(scores):
        return None
    df = map_imp(scores, names, raw)
    return None if df is None else list(df["feature"])

def perm_subset(m, Xdf, ys, cols, n_repeats):
    base = score_est(m, Xdf, ys)
    rng = np.random.RandomState(42)
    rows = []
    for c in cols:
        vals = []
        for _ in range(n_repeats):
            Xp = Xdf.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            vals.append(base - score_est(m, Xp, ys))
        rows.append((c, float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)))
    return pd.DataFrame(rows, columns=["feature", "importance_mean", "importance_sd"]).sort_values(
        "importance_mean", ascending=False
    )

n = len(feats)
if n <= 500:
    r = permutation_importance(model, X, y, scoring=score_est, n_repeats=20, random_state=42, n_jobs=1)
    imp = pd.DataFrame({
        "feature": feats,
        "importance_mean": r.importances_mean,
        "importance_sd": r.importances_std,
    }).sort_values("importance_mean", ascending=False)
    method, budget = "validation_permutation", f"n_features={n};n_repeats=20"
else:
    imp = model_importances(model, feats)
    if imp is not None:
        imp["importance_sd"] = 0.0
        method, budget = "model_importances", f"n_features={n};n_repeats=0"
    else:
        rank = selector_ranking(feats)
        if rank is None:
            rank = feats[:50]
            method = "permutation_top50_no_ranking"
        else:
            method = "permutation_top50"
        cols = rank[:50]
        imp = perm_subset(model, X, y, cols, 5)
        budget = f"n_features={n};permuted={len(cols)};n_repeats=5"

out = imp.head(15)[["feature", "importance_mean", "importance_sd"]].copy().round(6)
mlca_runtime.save_table(out, "interpretation.csv", caption=f"Frozen model feature importance; method={method}; {budget}")
if len(out):
    mlca_runtime.record_fact_candidate("top_feature", str(out.iloc[0]["feature"]), source="interpretation.csv")
print(f"train={train.shape[0]} val={val.shape[0]} n_model_inputs={n}")
print(f"method={method} budget={budget}")
print(f"top_feature={out.iloc[0]['feature'] if len(out) else 'NA'}")
