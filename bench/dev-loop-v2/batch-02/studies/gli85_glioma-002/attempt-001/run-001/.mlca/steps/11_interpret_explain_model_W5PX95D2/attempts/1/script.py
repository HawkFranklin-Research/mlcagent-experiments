import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
prot = json.loads(inp["_protocol"].read_text())
targets, excluded = prot["targets"], prot["excluded_columns"]
group = prot.get("group_column")
train = pd.read_parquet(inp["train"]); val = pd.read_parquet(inp["val"])
print(f"train {len(train)} val {len(val)}")

def load(k):
    with inp[k].open("rb") as f: return pickle.load(f)
pre = load("05_preprocess_fit_preprocessing_M7MKT30F/preprocessor.pkl")
sel = load("06_features_select_features_18G3F1C0/selector.pkl")
model = load("latest_model")
raw_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
raw_cols = [c for c in raw_cols if not ((train[c].dtype == object or isinstance(train[c].dtype, pd.CategoricalDtype)) and train[c].nunique(dropna=False) == len(train))]
def fni(o): return list(getattr(o, "feature_names_in_", []))
if isinstance(model, Pipeline): predictor = model
elif set(fni(model)) == set(raw_cols): predictor = model
elif set(fni(sel)) == set(raw_cols): predictor = Pipeline([("selector", sel), ("model", model)])
elif set(fni(pre)) == set(raw_cols): predictor = Pipeline([("preprocessor", pre), ("selector", sel), ("model", model)])
else: predictor = model
input_cols = list(getattr(predictor, "feature_names_in_", raw_cols))
if not set(input_cols).issubset(val.columns): raise ValueError("missing input columns")
Xv = val[input_cols].copy(); y = val[targets[0]]
classes = getattr(predictor, "classes_", np.unique(y))
def score(X):
    p = predictor.predict_proba(X) if hasattr(predictor, "predict_proba") else predictor.decision_function(X)
    if len(classes) == 2:
        s = p[:, 1] if getattr(p, "ndim", 1) > 1 else p
        return roc_auc_score((y == classes[1]).astype(int), s)
    return roc_auc_score(y, p, multi_class="ovr", average="macro", labels=classes)
base = score(Xv)
est = predictor.steps[-1][1] if isinstance(predictor, Pipeline) else predictor
vals = None
if hasattr(est, "coef_"):
    coef = np.asarray(est.coef_)
    vals = np.abs(coef.ravel()) if coef.ndim == 1 or coef.shape[0] == 1 else np.abs(coef).mean(axis=0)
elif hasattr(est, "feature_importances_"):
    vals = np.abs(est.feature_importances_)
rank = None
if vals is not None:
    try: names = predictor[:-1].get_feature_names_out() if isinstance(predictor, Pipeline) else getattr(est, "feature_names_in_", None)
    except Exception: names = getattr(est, "feature_names_in_", None)
    if names is not None and len(names) == len(vals):
        raw = list(getattr(predictor, "feature_names_in_", raw_cols))
        mapped = []
        for n in names:
            s = str(n); found = s if s in raw else None
            if found is None and "__" in s and s.split("__")[-1] in raw: found = s.split("__")[-1]
            if found is None:
                for r in raw:
                    if s.endswith("__"+str(r)) or s.endswith(str(r)): found = r; break
            if found is None: mapped = None; break
            mapped.append(found)
        if mapped is not None:
            rank = pd.Series(vals, index=mapped).groupby(level=0).sum().sort_values(ascending=False)
n_feat = len(input_cols)
if n_feat > 500:
    if rank is None:
        method = "unavailable"; out = pd.DataFrame([{"feature":"unavailable","importance_mean":np.nan,"importance_sd":np.nan}])
        top_feature = "unavailable"; n_perm = n_rep = 0
    else:
        feats = list(rank.index[:min(50, len(rank))]); n_rep = 5; n_perm = len(feats)
        rng = np.random.default_rng(42); rows = []
        for f in feats:
            drops = []
            for _ in range(n_rep):
                Xp = Xv.copy(); Xp[f] = rng.permutation(Xp[f].values)
                drops.append(base - score(Xp))
            rows.append({"feature": f, "importance_mean": np.mean(drops), "importance_sd": np.std(drops, ddof=1) if n_rep > 1 else 0.0})
        out = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).head(15)
        method = "permutation_top50_by_model_importance"; top_feature = out.iloc[0]["feature"]
else:
    n_rep = 20; n_perm = n_feat; rng = np.random.default_rng(42); rows = []
    for f in input_cols:
        drops = []
        for _ in range(n_rep):
            Xp = Xv.copy(); Xp[f] = rng.permutation(Xp[f].values)
            drops.append(base - score(Xp))
        rows.append({"feature": f, "importance_mean": np.mean(drops), "importance_sd": np.std(drops, ddof=1)})
    out = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).head(15)
    method = "permutation_all"; top_feature = out.iloc[0]["feature"]
out = out[["feature","importance_mean","importance_sd"]]
for c in ["importance_mean","importance_sd"]:
    out[c] = out[c].map(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)
mlca_runtime.save_table(out, "interpretation.csv", caption=f"method={method}; permuted_features={n_perm}; repeats={n_rep}; baseline_roc_auc={base:.4f}")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
mlca_runtime.record_metric("interpretation_feature_budget", float(n_perm), split="val")
mlca_runtime.record_metric("interpretation_repeats", float(n_rep), split="val")
print(f"method {method} permuted {n_perm} repeats {n_rep} baseline {base:.4f}")
print(f"top_feature {top_feature}")
