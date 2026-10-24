import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = set(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
task = protocol["task"]
n_classes = task.get("n_classes", 2)

val = pd.read_parquet(inputs["val"])
def load_pkl(k):
    with open(inputs[k], "rb") as f:
        return pickle.load(f)

pre = load_pkl("06_preprocess_fit_preprocessing_Z6Z720YG/preprocessor.pkl")
sel = load_pkl("07_features_select_features_Y69JECM8/selector.pkl")
model = load_pkl("latest_model")
if isinstance(model, Pipeline):
    pipe = model
else:
    steps = []
    if pre is not None: steps.append(("preprocessor", pre))
    if sel is not None: steps.append(("selector", sel))
    steps.append(("model", model))
    pipe = Pipeline(steps)

raw_features = None
if hasattr(pipe, "feature_names_in_"):
    raw_features = list(pipe.feature_names_in_)
elif hasattr(pipe, "named_steps"):
    first = next(iter(pipe.named_steps.values()))
    if hasattr(first, "feature_names_in_"):
        raw_features = list(first.feature_names_in_)
if raw_features is None:
    raw_features = [c for c in val.columns if c not in targets and c not in excluded and c != group]
X_val = val[raw_features].copy()
y_val = val[targets[0]]
y_val = pd.factorize(y_val)[0] if y_val.dtype == object else y_val.values

def extract_importance(pipe):
    est = pipe.named_steps[list(pipe.named_steps)[-1]] if hasattr(pipe, "named_steps") else pipe
    coef = None
    if hasattr(est, "coef_"):
        coef = np.abs(est.coef_)
        if coef.ndim > 1:
            coef = coef.mean(axis=0) if coef.shape[0] > 1 else coef[0]
    elif hasattr(est, "feature_importances_"):
        coef = est.feature_importances_
    if coef is None:
        return None
    if hasattr(pipe, "named_steps"):
        for step in pipe.named_steps.values():
            if hasattr(step, "get_support") and hasattr(step, "feature_names_in_"):
                mask = step.get_support()
                raw_names = list(step.feature_names_in_)
                if len(mask) == len(raw_names) and len(coef) == mask.sum():
                    return pd.Series(coef, index=np.array(raw_names)[mask])
        if hasattr(pipe, "feature_names_in_") and len(coef) == len(pipe.feature_names_in_):
            return pd.Series(coef, index=pipe.feature_names_in_)
    if hasattr(est, "feature_names_in_") and len(coef) == len(est.feature_names_in_):
        return pd.Series(coef, index=est.feature_names_in_)
    return None

imp = extract_importance(pipe)
n_input_features = len(raw_features)
scoring = "roc_auc" if n_classes == 2 else "roc_auc_ovr"

def score(X):
    proba = pipe.predict_proba(X)
    if n_classes == 2:
        return roc_auc_score(y_val, proba[:, 1])
    return roc_auc_score(y_val, proba, multi_class="ovr", average="macro")

if imp is not None:
    imp = imp.sort_values(ascending=False)
    if n_input_features <= 500:
        n_repeats, feats = 20, raw_features
    else:
        n_repeats, feats = 5, [f for f in imp.index if f in raw_features][:50]
        if not feats: feats = raw_features[:50]
    base = score(X_val)
    rng = np.random.RandomState(42)
    means, sds = [], []
    for feat in feats:
        sc = []
        col = X_val[feat].values
        for _ in range(n_repeats):
            Xp = X_val.copy()
            Xp[feat] = rng.permutation(col)
            sc.append(score(Xp))
        means.append(base - np.mean(sc))
        sds.append(np.std(sc))
    res = pd.DataFrame({"feature": feats, "importance_mean": means, "importance_sd": sds})
    res = res.sort_values("importance_mean", ascending=False).head(15)
else:
    if n_input_features <= 500:
        n_repeats, feats = 20, raw_features
        result = permutation_importance(pipe, X_val, y_val, scoring=scoring, n_repeats=n_repeats, random_state=42, n_jobs=1)
        res = pd.DataFrame({"feature": raw_features, "importance_mean": result.importances_mean, "importance_sd": result.importances_std})
        res = res.sort_values("importance_mean", ascending=False).head(15)
    else:
        n_repeats, feats = 0, []
        res = pd.DataFrame([{"feature": "LIMITATION_NO_RELIABLE_RANKING", "importance_mean": np.nan, "importance_sd": np.nan}])

mlca_runtime.save_table(res, "interpretation.csv", caption="Top features by permutation importance on validation.")
top = res.iloc[0]["feature"] if len(res) else "unavailable"
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
mlca_runtime.record_metric("n_input_features", float(n_input_features), "val")
mlca_runtime.record_metric("n_features_permuted", float(len(feats)), "val")
mlca_runtime.record_metric("n_repeats", float(n_repeats), "val")
print(f"permutation importance: features={n_input_features} permuted={len(feats)} repeats={n_repeats}")
print(f"top_feature={top}")
