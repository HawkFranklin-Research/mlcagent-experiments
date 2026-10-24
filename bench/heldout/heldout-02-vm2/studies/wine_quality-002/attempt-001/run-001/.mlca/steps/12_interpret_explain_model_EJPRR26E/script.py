import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column", None)
task = protocol["task"]
primary_metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
print(f"train={len(train)} val={len(val)}")

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)
feature_cols = [c for c in train.columns if c not in drop]

def is_keylike(name, s):
    if len(s) <= 1 or not s.is_unique:
        return False
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        return True
    if pd.api.types.is_integer_dtype(s.dtype) and any(t in name.lower() for t in ["id", "code", "key"]):
        return True
    return False

feature_cols = [c for c in feature_cols if not is_keylike(c, train[c])]

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_input_cols(obj):
    cols = getattr(obj, "feature_names_in_", None)
    if cols is None and hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                cols = step.feature_names_in_
                break
    return list(cols) if cols is not None else None

model = load_model(inputs["latest_model"])
model_cols = get_input_cols(model)
if model_cols is None or set(model_cols) != set(feature_cols):
    try:
        alt = load_model(inputs["08_model_train_logistic_DZS6CMQH/model.pkl"])
        alt_cols = get_input_cols(alt)
        if alt_cols is not None and set(alt_cols) == set(feature_cols):
            model, model_cols = alt, alt_cols
    except Exception:
        pass
if model_cols is None:
    model_cols = feature_cols
if set(model_cols) != set(feature_cols):
    raise ValueError(f"model input columns mismatch: model={model_cols} data={feature_cols}")

X = val[model_cols]
y = val[targets[0]].to_numpy()
scoring = "roc_auc_ovr" if primary_metric == "roc_auc_ovr" else "roc_auc"
n_features = len(model_cols)

if n_features <= 500:
    method = "validation_permutation_importance"
    n_repeats = 20
    pi = permutation_importance(
        model, X, y, scoring=scoring, n_repeats=n_repeats,
        random_state=42, n_jobs=1,
    )
    imp_mean = pi.importances_mean
    imp_sd = pi.importances_std
else:
    method = "model_native_importance"
    n_repeats = 0
    est = list(model.named_steps.values())[-1] if hasattr(model, "named_steps") else model
    if hasattr(est, "coef_"):
        native = np.abs(est.coef_)
        if native.ndim > 1:
            native = native.mean(axis=0)
    elif hasattr(est, "feature_importances_"):
        native = est.feature_importances_
    else:
        native = np.full(n_features, np.nan)
    if len(native) == n_features:
        imp_mean = native
        imp_sd = np.full(n_features, np.nan)
    else:
        imp_mean = np.full(n_features, np.nan)
        imp_sd = np.full(n_features, np.nan)

order = np.argsort(np.nan_to_num(imp_mean, nan=-np.inf))[::-1]
top_n = min(15, n_features)

def sig4(x):
    return np.nan if pd.isna(x) else float(f"{x:.4g}")

rows = [
    {
        "feature": model_cols[i],
        "importance_mean": sig4(imp_mean[i]),
        "importance_sd": sig4(imp_sd[i]),
    }
    for i in order[:top_n]
]
interp = pd.DataFrame(rows, columns=["feature", "importance_mean", "importance_sd"])
caption = f"method={method}; features={n_features}; repeats={n_repeats}; scoring={scoring}"
mlca_runtime.save_table(interp, "interpretation.csv", caption=caption)

top_feature = str(interp.iloc[0]["feature"]) if len(interp) else None
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"method={method} features={n_features} repeats={n_repeats} scoring={scoring}")
print(f"top_feature={top_feature} importance={interp.iloc[0]['importance_mean'] if len(interp) else np.nan}")
