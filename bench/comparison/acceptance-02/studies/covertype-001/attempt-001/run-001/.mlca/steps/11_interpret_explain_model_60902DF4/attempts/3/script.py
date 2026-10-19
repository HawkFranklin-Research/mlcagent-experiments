import json, pickle
import mlca_runtime
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
metric = protocol["task"]["primary_metric"]
group_col = protocol.get("group_column")
drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
val = pd.read_parquet(inp["val"])

def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def raw_cols():
    n = len(val)
    cols = []
    for c in val.columns:
        if c in drop:
            continue
        s = val[c]
        nun = s.nunique(dropna=True)
        if s.dtype.kind in "O" and nun == n:
            continue
        if pd.api.types.is_integer_dtype(s) and nun == n:
            continue
        cols.append(c)
    return cols

def names_of(obj):
    names = getattr(obj, "feature_names_in_", None)
    if names is not None:
        return list(names)
    if isinstance(obj, Pipeline):
        for _, step in obj.named_steps.items():
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                return list(names)
    return None

def usable(obj):
    if not hasattr(obj, "predict_proba"):
        return None
    names = names_of(obj) or raw_cols()
    if not set(names).issubset(val.columns) or set(names) & drop:
        return None
    try:
        obj.predict_proba(val[names].head(2))
        return names
    except Exception:
        return None

prep = load(inp["05_preprocess_fit_preprocessing_Z8NMCAXW/preprocessor.pkl"])
sel = load(inp["06_features_select_features_53HVANE8/selector.pkl"])
mod = load(inp["07_model_train_logistic_Z4Z606HV/model.pkl"])
cands = []
for key in ["latest_model", "07_model_train_logistic_Z4Z606HV/model.pkl",
            "08_model_train_alternative_JKAAJDG0/alternative_model.pkl"]:
    if key in inp:
        try:
            cands.append(load(inp[key]))
        except Exception:
            pass
cands += [
    Pipeline([("prep", prep), ("sel", sel), ("mod", mod)]),
    Pipeline([("sel", sel), ("mod", mod)]),
    Pipeline([("prep", prep), ("mod", mod)]),
    mod,
]

pipe, feature_cols = None, None
for obj in cands:
    names = usable(obj)
    if names is not None:
        pipe, feature_cols = obj, names
        break
if pipe is None:
    raise RuntimeError("No usable frozen predictor found")

X_val = val[feature_cols].copy()
y_val = val[targets[0]].copy()
mask = y_val.notna()
X_val, y_val = X_val[mask], y_val[mask]
result = permutation_importance(
    pipe, X_val, y_val, scoring=metric, n_repeats=20, random_state=42, n_jobs=1
)

imp = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": np.round(result.importances_mean, 4),
    "importance_sd": np.round(result.importances_std, 4),
}).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
mlca_runtime.save_table(imp, "interpretation.csv", caption="Validation permutation importance")
top_feature = str(imp.iloc[0]["feature"]) if len(imp) else ""
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"val_rows={len(X_val)} feats={len(feature_cols)} repeats=20")
print(f"top_feature={top_feature}")
