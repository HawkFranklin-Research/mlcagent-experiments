import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
print(f"train {train.shape}, val {val.shape}")

def is_keylike(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False

exc = set(targets) | set(excluded)
if group: exc.add(group)
feature_cols = [c for c in train.columns if c not in exc and not is_keylike(train[c])]
print(f"n_features={len(feature_cols)}")

with open(inputs["latest_model"], "rb") as f: latest_model = pickle.load(f)
with open(inputs["07_model_train_logistic_N3PQQ99D/model.pkl"], "rb") as f: primary_model = pickle.load(f)

def find_source(est, depth=0, seen=None):
    if seen is None: seen = set()
    if depth > 10 or id(est) in seen: return None
    seen.add(id(est))
    for a in ("coef_", "feature_importances_"):
        try:
            if getattr(est, a, None) is not None: return est
        except Exception: pass
    ch = []
    if isinstance(est, dict): ch = list(est.values())
    elif isinstance(est, (list, tuple)): ch = list(est)
    else:
        for a in ("steps","estimators_","calibrated_classifiers_","named_estimators_",
                  "estimator","final_estimator_","best_estimator_","base_estimator_",
                  "estimator_","model"):
            v = getattr(est, a, None)
            if v is None: continue
            if isinstance(v, dict): ch.extend(v.values())
            elif isinstance(v, (list, tuple)): ch.extend(v)
            else: ch.append(v)
    for c in ch:
        if isinstance(c, tuple) and len(c) == 2: c = c[1]
        r = find_source(c, depth+1, seen)
        if r is not None: return r
    return None

def extract(model):
    src = find_source(model)
    if src is None: return None
    if getattr(src, "coef_", None) is not None:
        c = np.asarray(src.coef_)
        imp = np.abs(c).mean(0) if c.ndim > 1 else np.abs(c)
        sd = np.abs(c).std(0) if c.ndim > 1 else np.zeros_like(imp)
        kind = "coef"
    else:
        imp = np.asarray(src.feature_importances_); sd = np.zeros_like(imp); kind = "fi"
    names = None
    fn = getattr(src, "feature_names_in_", None)
    if fn is not None and len(fn) == len(imp): names = list(fn)
    if names is None and hasattr(model, "get_feature_names_out"):
        try:
            g = list(model.get_feature_names_out())
            if len(g) == len(imp): names = g
        except Exception: pass
    if names is None and hasattr(model, "steps"):
        for i in range(len(model.steps)-1, 0, -1):
            try:
                g = list(model[:i].get_feature_names_out())
                if len(g) == len(imp): names = g; break
            except Exception: pass
    return imp, sd, names, kind

r_latest = extract(latest_model)
r_primary = extract(primary_model)

def save_result(imp, sd, names, kind, tag, extra=""):
    order = np.argsort(-np.asarray(imp))[:15]
    feats = [names[i] if names is not None else f"feature_{int(i)}" for i in order]
    res = pd.DataFrame({
        "feature": feats,
        "importance_mean": np.round(np.asarray(imp)[order].astype(float), 4),
        "importance_sd": np.round(np.asarray(sd)[order].astype(float), 4),
    })
    cap = f"method={tag}_{kind}{extra}; ranked={len(imp)}"
    mlca_runtime.save_table(res, "interpretation.csv", caption=cap)
    mlca_runtime.record_fact_candidate("top_feature", res.iloc[0]["feature"], source="interpretation.csv")
    print(f"method={tag}_{kind}{extra}; ranked={len(imp)}; top={res.iloc[0]['feature']}")

perm_done = False
if r_latest is None and r_primary is not None and r_primary[2] is not None and len(feature_cols) > 500:
    imp_p, _, names_p, _ = r_primary
    if len(names_p) == len(imp_p):
        top50 = [n for n in [names_p[i] for i in np.argsort(-imp_p)[:50]] if n in val.columns]
        model_feats = list(getattr(latest_model, "feature_names_in_", feature_cols))
        if top50 and model_feats and all(c in val.columns for c in model_feats):
            try:
                from sklearn.metrics import roc_auc_score
                X_val = val[model_feats]; y_val = val[targets[0]]
                def sc(X):
                    return roc_auc_score(y_val, latest_model.predict_proba(X),
                                         multi_class="ovr", average="macro")
                base = sc(X_val); rng = np.random.RandomState(42); reps = 5
                vals = np.zeros((len(top50), reps))
                for i, f in enumerate(top50):
                    for r in range(reps):
                        Xp = X_val.copy(); col = Xp[f].to_numpy().copy()
                        rng.shuffle(col); Xp[f] = col
                        vals[i, r] = base - sc(Xp)
                save_result(vals.mean(1), vals.std(1), top50, "perm_val", "latest",
                            f"; n_repeats={reps}; ranker=primary_coef")
                perm_done = True
            except Exception as e:
                print(f"perm_failed: {type(e).__name__}")

if not perm_done:
    if r_latest is not None:
        save_result(*r_latest, "latest")
    elif r_primary is not None:
        save_result(*r_primary, "primary", "; frozen_latest_lacks_intrinsic_importance")
    else:
        res = pd.DataFrame({"feature": ["__limitation__"],
                            "importance_mean": [float('nan')],
                            "importance_sd": [float('nan')]})
        mlca_runtime.save_table(res, "interpretation.csv",
                                caption="limitation=no_reliable_importance_ranking")
        mlca_runtime.record_fact_candidate("top_feature", "unavailable",
                                           source="interpretation.csv")
        print("limitation; no importance ranking")
