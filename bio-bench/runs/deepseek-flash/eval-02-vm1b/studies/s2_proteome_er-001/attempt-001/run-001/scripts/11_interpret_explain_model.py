import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
excluded = set(protocol.get("excluded_columns") or [])
group = protocol.get("group_column")
task = protocol["task"]
metric = str(task.get("primary_metric", "roc_auc")).lower()
n_classes = int(task.get("n_classes", 2))

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
print(f"train={train.shape[0]} val={val.shape[0]}")

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
drop_key = [c for c in feature_cols if (train[c].dtype == object or pd.api.types.is_string_dtype(train[c].dtype)) and train[c].nunique(dropna=True) == len(train)]
feature_cols = [c for c in feature_cols if c not in drop_key]
target_col = targets[0]
y_val = val[target_col].to_numpy()

with open(inp["latest_model"], "rb") as f:
    model = pickle.load(f)
if isinstance(model, dict) and "model" in model:
    model = model["model"]

def get_input_cols(m):
    for obj in [m] + list(getattr(m, "named_steps", {}).values()):
        if hasattr(obj, "feature_names_in_"):
            return list(obj.feature_names_in_)
    return list(feature_cols)

model_cols = get_input_cols(model)
missing = [c for c in model_cols if c not in val.columns]
if missing:
    raise KeyError(f"Missing frozen model input columns: {missing}")
X_full = val[model_cols].copy()

def scorer(est, X, y):
    p = est.predict_proba(X)
    if n_classes == 2:
        return roc_auc_score(y, p[:, 1])
    return roc_auc_score(y, p, multi_class="ovr", average="macro", labels=np.unique(y))

def coeff_rank(m, cols):
    final = m.named_steps[list(m.named_steps)[-1]] if hasattr(m, "named_steps") else m
    if hasattr(final, "coef_"):
        co = np.asarray(final.coef_)
        if co.ndim == 1:
            imp = np.abs(co)
        elif co.shape[0] == 1:
            imp = np.abs(co[0])
        else:
            imp = np.abs(co).mean(axis=0)
    elif hasattr(final, "feature_importances_"):
        imp = np.abs(np.asarray(final.feature_importances_))
    else:
        return None
    if len(imp) == len(cols):
        return imp
    names = None
    try:
        pre = m[:-1] if hasattr(m, "__getitem__") else None
        if pre is not None:
            names = list(pre.get_feature_names_out())
    except Exception:
        names = None
    if (names is None or len(names) != len(imp)) and hasattr(m, "named_steps"):
        steps = list(m.named_steps.items())
        for i, (_, obj) in enumerate(steps):
            if hasattr(obj, "get_support"):
                try:
                    support = obj.get_support()
                except Exception:
                    continue
                before = None
                for _, obj2 in steps[:i]:
                    if hasattr(obj2, "get_feature_names_out"):
                        try:
                            before = list(obj2.get_feature_names_out())
                            break
                        except Exception:
                            pass
                if before is not None and len(before) == len(support):
                    names = [before[j] for j in range(len(before)) if support[j]]
                    break
    if names is None or len(names) != len(imp):
        return None
    d = {}
    for nm, v in zip(names, imp):
        found = None
        for cand in [nm, nm.split("__")[-1]]:
            if cand in cols:
                found = cand
                break
        if found is None:
            for rc in cols:
                if nm.endswith(rc):
                    found = rc
                    break
        if found is None:
            return None
        d[found] = d.get(found, 0.0) + float(v)
    return np.array([d.get(c, 0.0) for c in cols])

n_feat = len(model_cols)
if n_feat <= 500:
    method, reps = "permutation_full", 20
    res = permutation_importance(model, X_full, y_val, scoring=scorer, n_repeats=reps, random_state=42, n_jobs=1)
    imp = pd.DataFrame({"feature": model_cols, "importance_mean": res.importances_mean, "importance_sd": res.importances_std})
else:
    rank_imp = coeff_rank(model, model_cols)
    if rank_imp is None:
        method, reps = "unavailable", 0
        imp = pd.DataFrame({"feature": ["unavailable"], "importance_mean": [np.nan], "importance_sd": [np.nan]})
    else:
        method, reps = "coeff_top50_perm", 5
        top_idx = np.argsort(rank_imp)[::-1][:50]
        perm_cols = [model_cols[i] for i in top_idx]
        base = scorer(model, X_full, y_val)
        rng = np.random.default_rng(42)
        means, sds = [], []
        for col in perm_cols:
            orig = X_full[col].to_numpy(copy=True)
            vals = []
            for _ in range(reps):
                Xp = X_full.copy()
                Xp[col] = rng.permutation(orig)
                vals.append(base - scorer(model, Xp, y_val))
            means.append(float(np.mean(vals)))
            sds.append(float(np.std(vals, ddof=1) if reps > 1 else 0.0))
        imp = pd.DataFrame({"feature": perm_cols, "importance_mean": means, "importance_sd": sds})

if method != "unavailable":
    imp = imp.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
    top_feature = str(imp.loc[0, "feature"])
else:
    top_feature = "unavailable"

def sig4(x):
    if pd.isna(x) or float(x) == 0:
        return x
    return float(f"{float(x):.4g}")

for c in ["importance_mean", "importance_sd"]:
    imp[c] = imp[c].map(sig4)

mlca_runtime.save_table(imp, "interpretation.csv", caption="Top model feature importances")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"method={method} features={n_feat} repeats={reps} top={top_feature}")
