import mlca_runtime
import json
import pickle
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

excluded_set = set(targets) | set(excluded)
if group:
    excluded_set.add(group)
feature_cols = [
    c for c in train.columns
    if c not in excluded_set and not is_keylike(train[c])
]

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)
raw_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else feature_cols

final = model.steps[-1][1] if hasattr(model, "steps") else model
if hasattr(final, "coef_"):
    coefs = np.asarray(final.coef_)
    if coefs.ndim == 1:
        imp = np.abs(coefs)
        sd = np.zeros_like(imp)
    else:
        abscoef = np.abs(coefs)
        imp = abscoef.mean(axis=0)
        sd = abscoef.std(axis=0)
    method = "coefficients"
elif hasattr(final, "feature_importances_"):
    imp = np.asarray(final.feature_importances_)
    sd = np.zeros_like(imp)
    method = "feature_importances"
else:
    raise RuntimeError("frozen model has no coefficients or feature_importances_")

names = None
if hasattr(model, "feature_names_in_") and len(model.feature_names_in_) == len(imp):
    names = list(model.feature_names_in_)
elif len(raw_features) == len(imp):
    names = list(raw_features)
else:
    if hasattr(model, "get_feature_names_out"):
        try:
            names = list(model.get_feature_names_out())
        except Exception:
            pass
    if (names is None or len(names) != len(imp)) and hasattr(model, "steps") and len(model.steps) >= 2:
        prev = model.steps[-2][1]
        if hasattr(prev, "get_feature_names_out"):
            try:
                names = list(prev.get_feature_names_out())
            except Exception:
                pass
if names is None or len(names) != len(imp):
    raise RuntimeError(f"cannot map {len(imp)} importances to feature names")

res = pd.DataFrame({
    "feature": names,
    "importance_mean": imp,
    "importance_sd": sd,
}).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
res = res.round(4)

caption = (
    f"method={method}; frozen_model=latest_model; "
    f"features_ranked={len(imp)}; repeats=0; top15_shown"
)
mlca_runtime.save_table(res, "interpretation.csv", caption=caption)
mlca_runtime.record_fact_candidate(
    "top_feature", res.iloc[0]["feature"], source="interpretation.csv"
)
print(f"method={method}; ranked={len(imp)}; top={res.iloc[0]['feature']}")
