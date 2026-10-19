import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
if "group_column" not in protocol:
    raise KeyError(f"Missing 'group_column'; available keys: {list(protocol.keys())}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]
task = protocol["task"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
drop = set(targets) | set(excluded)
if group_col is not None:
    drop.add(group_col)

def is_key_like(s):
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.nunique(dropna=False) == len(s)
    return False

rule_cols = [c for c in train.columns if c not in drop and not is_key_like(train[c])]

candidates = [
    ("latest_model", inputs["latest_model"]),
    ("08_model_train_alternative_M1N0MCXA/alternative_model.pkl", inputs["08_model_train_alternative_M1N0MCXA/alternative_model.pkl"]),
    ("07_model_train_logistic_N8YSRX76/model.pkl", inputs["07_model_train_logistic_N8YSRX76/model.pkl"]),
    ("06_features_select_features_BD9SGMPY/selector.pkl", inputs["06_features_select_features_BD9SGMPY/selector.pkl"]),
    ("05_preprocess_fit_preprocessing_8FQ40592/preprocessor.pkl", inputs["05_preprocess_fit_preprocessing_8FQ40592/preprocessor.pkl"]),
]
chosen = None
chosen_name = None
for name, path in candidates:
    try:
        obj = pickle.loads(path.read_bytes())
    except Exception:
        continue
    if not hasattr(obj, "predict_proba") or not hasattr(obj, "feature_names_in_"):
        continue
    if list(obj.feature_names_in_) == rule_cols:
        chosen = obj
        chosen_name = name
        break

if chosen is None:
    for name, path in candidates:
        try:
            obj = pickle.loads(path.read_bytes())
        except Exception:
            continue
        if hasattr(obj, "predict_proba") and hasattr(obj, "feature_names_in_"):
            fn = list(obj.feature_names_in_)
            if all(c in train.columns for c in fn):
                chosen = obj
                chosen_name = name
                break

if chosen is None:
    raise RuntimeError("No complete saved pipeline with predict_proba and raw feature_names_in_ found")

feature_cols = list(chosen.feature_names_in_)
X_val = val[feature_cols].copy()
y_val = val[targets[0]] if len(targets) == 1 else val[targets].copy()
scoring = "roc_auc" if task.get("type") == "binary" else "roc_auc_ovr"
r = permutation_importance(chosen, X_val, y_val, scoring=scoring, n_repeats=20, random_state=42, n_jobs=1)

def round_sig(x, sig=4):
    if pd.isna(x) or x == 0:
        return x
    return float(f"{x:.{sig}g}")

imp = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": r.importances_mean,
    "importance_sd": r.importances_std,
})
imp = imp.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
imp["importance_mean"] = imp["importance_mean"].map(round_sig)
imp["importance_sd"] = imp["importance_sd"].map(round_sig)
mlca_runtime.save_table(imp, "interpretation.csv", caption="Permutation importance on validation rows")
top_feature = imp.iloc[0]["feature"] if len(imp) else None
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"train rows: {len(train)}")
print(f"val rows: {len(val)}")
print(f"pipeline: {chosen_name}")
print(f"top feature: {top_feature}")
print(f"interpretation rows: {len(imp)}")
