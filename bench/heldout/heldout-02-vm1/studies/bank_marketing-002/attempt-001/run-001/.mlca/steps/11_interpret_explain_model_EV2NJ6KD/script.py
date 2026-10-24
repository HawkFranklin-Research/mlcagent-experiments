import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for k in ["targets", "excluded_columns", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing _protocol[{k!r}]; available keys: {list(protocol)}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
task = protocol["task"]
group = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
model = pickle.loads(inputs["07_model_train_logistic_NVNX9XZ7/model.pkl"].read_bytes())
print(f"train {train.shape[0]} rows, val {val.shape[0]} rows")

def is_key_like(s: pd.Series) -> bool:
    return len(s) > 1 and s.nunique(dropna=False) == len(s) and (
        pd.api.types.is_integer_dtype(s) or pd.api.types.is_object_dtype(s)
    ) and not pd.api.types.is_float_dtype(s)

allowed = [
    c for c in train.columns
    if c not in set(targets) | set(excluded) | ({group} if group else set())
    and not is_key_like(train[c])
]
model_cols = list(getattr(model, "feature_names_in_", allowed))
feature_cols = [c for c in model_cols if c in val.columns]
if any(c in set(targets) | set(excluded) | ({group} if group else set()) for c in feature_cols):
    raise ValueError("Feature schema includes a target or excluded column")

X_val = val.loc[:, feature_cols].copy()
classes = getattr(model, "classes_", np.sort(val[target].unique()))
y_val = val[target].map({c: i for i, c in enumerate(classes)}).astype(int)
n_classes = int(task.get("n_classes", len(classes)))
scoring = "roc_auc" if n_classes <= 2 else "roc_auc_ovr"
print(f"permuting {len(feature_cols)} features with scoring={scoring}")

n_repeats = 5 if len(feature_cols) > 500 else 20
if len(feature_cols) > 500:
    est = model.named_steps[list(model.named_steps)[-1]] if hasattr(model, "named_steps") else model
    if hasattr(est, "coef_"):
        imp = np.abs(np.ravel(est.coef_))
    else:
        imp = getattr(est, "feature_importances_", None)
    if imp is None or len(imp) != len(feature_cols):
        raise ValueError("No reliable native importance ranking for >500 features; refusing full-width permutation")
    method = "native_importance"
    n_repeats = 0
    interp = pd.DataFrame({"feature": feature_cols, "importance_mean": imp, "importance_sd": np.nan})
else:
    method = "validation_permutation_importance"
    result = permutation_importance(
        model, X_val, y_val, scoring=scoring, n_repeats=n_repeats,
        random_state=42, n_jobs=1,
    )
    interp = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": result.importances_mean,
        "importance_sd": result.importances_std,
    })

interp = interp.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
for c in ["importance_mean", "importance_sd"]:
    interp[c] = interp[c].map(lambda x: float(f"{x:.4g}") if pd.notna(x) else x)
top_feature = str(interp.iloc[0]["feature"])
mlca_runtime.save_table(
    interp, "interpretation.csv",
    caption=f"method={method}; n_features={len(feature_cols)}; n_repeats={n_repeats}; scoring={scoring}",
)
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
mlca_runtime.record_metric("permutation_n_features", len(feature_cols), "val")
mlca_runtime.record_metric("permutation_n_repeats", n_repeats, "val")
print(f"top feature: {top_feature} ({interp.iloc[0]['importance_mean']})")
