import mlca_runtime, json, pickle
import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
P = mlca_runtime.inputs()
proto = json.loads(P["_protocol"].read_text())
targets = proto["targets"]
excluded = proto.get("excluded_columns", [])
group = proto.get("group_column")
metric = proto["task"]["primary_metric"]

train = pd.read_parquet(P["train"])
val = pd.read_parquet(P["val"])

skip = set(targets) | set(excluded)
if group:
    skip.add(group)
feat_cols = []
for c in train.columns:
    if c in skip:
        continue
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and \
       s.nunique(dropna=False) == len(s) and len(s) > 1:
        continue
    if pd.api.types.is_integer_dtype(s) and s.is_unique and len(s) > 1:
        u = np.sort(s.dropna().unique())
        if len(u) > 1 and np.all(np.diff(u) == 1):
            continue
    feat_cols.append(c)

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load(P["05_preprocess_fit_preprocessing_G69EDXSB/preprocessor.pkl"])
selector = load(P["06_features_select_features_32CBHHGS/selector.pkl"])
model = load(P["07_model_train_logistic_Z7HS8D6N/model.pkl"])
alt = load(P["08_model_train_alternative_MG1H66HT/alternative_model.pkl"])
latest = load(P["latest_model"])

def first_names(obj):
    if hasattr(obj, "steps") and obj.steps:
        fn = getattr(obj.steps[0][1], "feature_names_in_", None)
        if fn is not None:
            return list(fn)
    fn = getattr(obj, "feature_names_in_", None)
    return list(fn) if fn is not None else None

def raw_names_ok(obj):
    return first_names(obj) == feat_cols

def raw_ok(obj):
    return hasattr(obj, "predict_proba") and raw_names_ok(obj)

def raw_transform_ok(obj):
    return hasattr(obj, "transform") and raw_names_ok(obj)

cands = []
for lab, obj in [("latest", latest), ("alt", alt), ("model", model), ("selector", selector)]:
    if raw_ok(obj):
        cands.append((lab, obj))
if raw_transform_ok(selector) and hasattr(model, "predict_proba") and not raw_ok(model):
    cands.append(("selector+model", Pipeline([("selector", selector), ("model", model)])))
if raw_transform_ok(pre) and hasattr(selector, "transform") and not raw_names_ok(selector) \
   and hasattr(model, "predict_proba") and not raw_ok(model):
    cands.append(("pre+selector+model", Pipeline([("preprocessor", pre), ("selector", selector), ("model", model)])))

chosen = None
for lab, obj in cands:
    try:
        obj.predict_proba(val[feat_cols].iloc[:min(10, len(val))])
        chosen = (lab, obj)
        break
    except Exception:
        pass
if chosen is None:
    try:
        pipe = Pipeline([("preprocessor", pre), ("selector", selector), ("model", model)])
        pipe.predict_proba(val[feat_cols].iloc[:min(10, len(val))])
        chosen = ("fallback", pipe)
    except Exception as e:
        raise RuntimeError(f"No complete raw-feature pipeline could be used: {e}")

lab, est = chosen
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
y = val[targets[0]]
mask = y.notna()
X = val.loc[mask, feat_cols]
y = y.loc[mask]
print(f"train rows={len(train)}, val rows={len(val)}, features={len(feat_cols)}")
print(f"scoring={metric}, repeats=20, pipeline={lab}")

res = permutation_importance(est, X, y, scoring=metric, n_repeats=20, random_state=42, n_jobs=1)
imp = pd.DataFrame({
    "feature": feat_cols,
    "importance_mean": res.importances_mean,
    "importance_sd": res.importances_std,
})
imp = imp.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
for c in ["importance_mean", "importance_sd"]:
    imp[c] = imp[c].map(lambda v: float(f"{v:.4g}"))
mlca_runtime.save_table(imp, "interpretation.csv", caption="Permutation importance on validation rows")
top = str(imp.iloc[0]["feature"])
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
print(f"top feature={top}")
