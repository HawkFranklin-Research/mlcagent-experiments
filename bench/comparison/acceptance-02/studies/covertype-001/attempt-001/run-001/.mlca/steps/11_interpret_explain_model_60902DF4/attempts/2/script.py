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
excluded = list(protocol["excluded_columns"])
metric = protocol["task"]["primary_metric"]
group_col = protocol.get("group_column")
drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)

val = pd.read_parquet(inp["val"])

def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def probe(obj):
    if not hasattr(obj, "predict_proba"):
        return None
    names = getattr(obj, "feature_names_in_", None)
    if names is None:
        return None
    names = list(names)
    if not set(names).issubset(set(val.columns)) or set(names) & drop:
        return None
    try:
        _ = obj.predict_proba(val[names].head(2))
        return names
    except Exception:
        return None

pipe = None
feature_cols = None
for key in ["latest_model", "07_model_train_logistic_Z4Z606HV/model.pkl"]:
    obj = load(inp[key])
    names = probe(obj)
    if names is not None:
        pipe = obj
        feature_cols = names
        break

if pipe is None:
    prep = load(inp["05_preprocess_fit_preprocessing_Z8NMCAXW/preprocessor.pkl"])
    sel = load(inp["06_features_select_features_53HVANE8/selector.pkl"])
    mod = load(inp["07_model_train_logistic_Z4Z606HV/model.pkl"])
    pipe = Pipeline([("prep", prep), ("sel", sel), ("mod", mod)])
    feature_cols = list(prep.feature_names_in_)
    _ = pipe.predict_proba(val[feature_cols].head(2))

X_val = val[feature_cols].copy()
y_val = val[targets[0]].copy()
mask = y_val.notna()
X_val = X_val[mask]
y_val = y_val[mask]

result = permutation_importance(
    pipe, X_val, y_val, scoring=metric,
    n_repeats=20, random_state=42, n_jobs=1,
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
