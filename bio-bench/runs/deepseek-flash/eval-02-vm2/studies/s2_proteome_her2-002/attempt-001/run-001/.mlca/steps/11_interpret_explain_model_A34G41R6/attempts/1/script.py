import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
prot = json.loads(inp["_protocol"].read_text())
targets = prot["targets"]
excl_raw = prot.get("excluded_columns", [])
excl = set(excl_raw.get("examples", [])) if isinstance(excl_raw, dict) else set(excl_raw)
group = prot.get("group_column")
task = prot["task"]
metric = task.get("primary_metric", "roc_auc")
n_classes = int(task.get("n_classes", 2))

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
y_col = targets[0]
y_val = val[y_col].to_numpy()
print(f"train rows={len(train)} val rows={len(val)}")

def key_like(s):
    if pd.api.types.is_numeric_dtype(s):
        return False
    return s.nunique(dropna=False) == len(s)

feature_columns = [c for c in train.columns
                   if c not in set(targets)
                   and c not in excl
                   and not (group and c == group)
                   and not key_like(train[c])]

model_path = inp.get("latest_model") or inp["07_model_train_logistic_MCZZ4993/model.pkl"]
with open(model_path, "rb") as f:
    model = pickle.load(f)

def input_names(m):
    if hasattr(m, "feature_names_in_") and len(m.feature_names_in_) > 0:
        return list(m.feature_names_in_)
    if hasattr(m, "named_steps"):
        for step in m.named_steps.values():
            if hasattr(step, "feature_names_in_") and len(step.feature_names_in_) > 0:
                return list(step.feature_names_in_)
    return None

model_features = input_names(model) or feature_columns
missing = [c for c in model_features if c not in val.columns]
if missing:
    raise KeyError(f"Missing model features in val: {missing[:5]}")
if y_col not in val.columns:
    raise KeyError(f"Missing target column: {y_col}")
Xval = val[model_features].copy()

def score_df(df):
    p = model.predict_proba(df[model_features])
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    if p.shape[1] == 2:
        cls = getattr(model, "classes_", None)
        if cls is None and hasattr(model, "named_steps"):
            cls = getattr(list(model.named_steps.values())[-1], "classes_", None)
        pos = cls.index(1) if cls is not None and 1 in list(cls) else 1
        return roc_auc_score(y_val, p[:, pos])
    return roc_auc_score(y_val, p, multi_class="ovr", average="macro")

def model_importances(m):
    steps = list(m.named_steps.items()) if hasattr(m, "named_steps") else [("model", m)]
    last = steps[-1][1]
    imp = getattr(last, "feature_importances_", None)
    if imp is None and hasattr(last, "coef_"):
        coef = np.asarray(last.coef_)
        imp = np.abs(coef).ravel() if coef.ndim == 1 else np.abs(coef).mean(axis=0)
    if imp is None:
        return None
    names = list(last.feature_names_in_) if hasattr(last, "feature_names_in_") else None
    if names is None or len(names) != len(imp):
        for _, step in steps:
            if hasattr(step, "get_feature_names_out"):
                try:
                    names = list(step.get_feature_names_out())
                    break
                except Exception:
                    pass
    if names is None or len(names) != len(imp):
        return None
    s = pd.Series(imp, index=names)
    s = s[[c for c in s.index if c in model_features]]
    return s if len(s) else None

n_feat = len(model_features)
if n_feat <= 500:
    perm_feats = list(model_features)
    n_repeats = 20
    method = "full_validation_permutation"
else:
    imp = model_importances(model)
    if imp is not None:
        perm_feats = imp.abs().sort_values(ascending=False).head(50).index.tolist()
        method = "top50_model_importance_validation_permutation"
    else:
        corr = Xval.apply(lambda s: abs(s.corr(pd.Series(y_val, index=Xval.index)))).fillna(0)
        perm_feats = corr.sort_values(ascending=False).head(50).index.tolist()
        method = "top50_correlation_validation_permutation_limitation"
    perm_feats = [f for f in perm_feats if f in model_features]
    n_repeats = 5

base = score_df(Xval)
rows = []
for i, feat in enumerate(perm_feats):
    diffs = []
    orig = Xval[feat].to_numpy(copy=True)
    for r in range(n_repeats):
        rng = np.random.default_rng(42 + 1000 * i + r)
        Xp = Xval.copy(deep=False)
        Xp[feat] = rng.permutation(orig)
        diffs.append(base - score_df(Xp))
    sd = float(np.std(diffs, ddof=1)) if n_repeats > 1 else 0.0
    rows.append((feat, float(np.mean(diffs)), sd))

interp = (pd.DataFrame(rows, columns=["feature", "importance_mean", "importance_sd"])
          .sort_values("importance_mean", ascending=False)
          .head(15)
          .reset_index(drop=True))
for col in ["importance_mean", "importance_sd"]:
    interp[col] = interp[col].map(lambda x: float(f"{x:.4g}") if pd.notna(x) else x)

mlca_runtime.save_table(interp, "interpretation.csv", caption="Validation permutation feature importance")
top = str(interp.iloc[0]["feature"])
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
mlca_runtime.record_metric("top_feature_importance_mean", float(interp.iloc[0]["importance_mean"]), split="val")
print(f"model_features={n_feat} permuted={len(perm_feats)} repeats={n_repeats} method={method}")
print(f"top_feature={top}")
