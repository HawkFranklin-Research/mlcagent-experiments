import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
task = protocol["task"]
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)

def is_key_like(s: pd.Series, name: str) -> bool:
    if pd.api.types.is_numeric_dtype(s.dtype):
        return False
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == len(s)
    return False

feature_cols = [c for c in train.columns if c not in drop and not is_key_like(train[c], c)]

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)
if isinstance(model, dict):
    for k in ("model", "pipeline", "estimator"):
        if k in model:
            model = model[k]
            break

def final_estimator(m):
    return m.named_steps[m.steps[-1][0]] if isinstance(m, Pipeline) else m

def names_for_importance(m, n, feature_cols):
    final = final_estimator(m)
    if hasattr(final, "feature_names_in_") and len(final.feature_names_in_) == n:
        return list(final.feature_names_in_)
    if isinstance(m, Pipeline):
        try:
            names = m[:-1].get_feature_names_out()
            if len(names) == n:
                return list(names)
        except Exception:
            pass
        for _, step in m.named_steps.items():
            if hasattr(step, "get_support") and hasattr(step, "feature_names_in_"):
                try:
                    sup = step.get_support()
                    fn = np.asarray(step.feature_names_in_)
                    if len(sup) == len(fn) and int(np.sum(sup)) == n:
                        return list(fn[sup])
                except Exception:
                    pass
        if hasattr(m, "feature_names_in_") and len(m.feature_names_in_) == n:
            return list(m.feature_names_in_)
    if n == len(feature_cols):
        return list(feature_cols)
    return list(feature_cols[:n])

final = final_estimator(model)
coef = getattr(final, "coef_", None)
if coef is not None:
    coef = np.asarray(coef)
    if coef.ndim == 1:
        imp = np.abs(coef)
        imp_sd = np.full(imp.shape, np.nan)
    else:
        a = np.abs(coef)
        imp = a.mean(axis=0)
        imp_sd = a.std(axis=0)
    method = "coefficient_importance"
else:
    fi = getattr(final, "feature_importances_", None)
    if fi is None:
        out = pd.DataFrame(columns=["feature", "importance_mean", "importance_sd"])
        mlca_runtime.save_table(out, "interpretation.csv", caption="method=none; limitation=no reliable feature importance")
        print(f"train={len(train)} val={len(val)} features={len(feature_cols)}")
        print("No reliable feature importance ranking available.")
        raise SystemExit(0)
    imp = np.asarray(fi)
    imp_sd = np.full(imp.shape, np.nan)
    method = "feature_importances"

names = names_for_importance(model, len(imp), feature_cols)
if len(names) != len(imp):
    names = list(feature_cols[:len(imp)])

order = np.argsort(imp)[::-1][:15]
out = pd.DataFrame({
    "feature": [names[i] for i in order],
    "importance_mean": imp[order],
    "importance_sd": imp_sd[order],
})
for col in ("importance_mean", "importance_sd"):
    out[col] = out[col].map(lambda v: float(f"{v:.4g}") if pd.notna(v) else np.nan)

caption = f"method={method}; features_ranked={len(imp)}; repeats=0; top_k={len(out)}"
mlca_runtime.save_table(out, "interpretation.csv", caption=caption)
mlca_runtime.record_fact_candidate("top_feature", str(out.iloc[0]["feature"]), source="interpretation.csv")

print(f"train={len(train)} val={len(val)} features={len(feature_cols)} ranked={len(imp)}")
print(caption)
print(f"top_feature={out.iloc[0]['feature']}")
