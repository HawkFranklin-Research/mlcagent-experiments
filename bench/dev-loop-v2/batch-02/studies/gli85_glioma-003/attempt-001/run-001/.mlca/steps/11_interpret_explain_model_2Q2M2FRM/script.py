import mlca_runtime, json, pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"] if "group_column" in protocol else None
drop = set(targets) | set(excluded) | ({group} if group is not None else set())

train = pd.read_parquet(inputs["train"])
feature_cols = []
for c in [c for c in train.columns if c not in drop]:
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
        u = np.sort(s.dropna().unique())
        if len(u) == len(s) and u[0] in (0, 1) and u[-1] - u[0] + 1 == len(u):
            continue
    feature_cols.append(c)

model = pickle.loads(inputs["latest_model"].read_bytes())
est = list(model.named_steps.values())[-1] if hasattr(model, "named_steps") else model

if hasattr(est, "coef_"):
    coef = np.asarray(est.coef_)
    if coef.ndim > 1:
        coef = coef[0] if coef.shape[0] == 1 else np.abs(coef).mean(axis=0)
    importance = np.abs(np.asarray(coef).ravel())
    method = "frozen model coefficients"
elif hasattr(est, "feature_importances_"):
    importance = np.asarray(est.feature_importances_, dtype=float).ravel()
    method = "frozen model feature_importances_"
else:
    importance = None
    method = "unavailable"

names = None
if importance is not None and hasattr(model, "named_steps"):
    try:
        cand = np.asarray(model[:-1].get_feature_names_out(), dtype=str)
        if len(cand) == len(importance):
            names = cand
    except Exception:
        pass
    if names is None:
        for obj in model.named_steps.values():
            if hasattr(obj, "get_feature_names_out"):
                try:
                    cand = np.asarray(obj.get_feature_names_out(), dtype=str)
                    if len(cand) == len(importance):
                        names = cand
                        break
                except Exception:
                    pass
if importance is not None and names is None and hasattr(model, "feature_names_in_") and len(model.feature_names_in_) == len(importance):
    names = np.asarray(model.feature_names_in_, dtype=str)
if importance is not None and names is None and len(feature_cols) == len(importance):
    names = np.asarray(feature_cols, dtype=str)

if importance is None or names is None:
    out = pd.DataFrame(columns=["feature", "importance_mean", "importance_sd"])
    mlca_runtime.save_table(
        out,
        "interpretation.csv",
        caption="No reliable frozen-model importance ranking was available; full-width permutation was not performed.",
    )
    print(f"train rows={len(train)}; total feature cols={len(feature_cols)}; ranking unavailable")
    raise SystemExit(0)

order = np.argsort(-importance)[:15]
out = pd.DataFrame({
    "feature": names[order].astype(str),
    "importance_mean": importance[order],
    "importance_sd": np.zeros(len(order)),
})
for c in ["importance_mean", "importance_sd"]:
    out[c] = out[c].map(lambda x: float(f"{x:.4g}"))

caption = f"Method: {method}; ranking budget: {len(importance)} features; permutation repeats: 0 (n_features>500)."
mlca_runtime.save_table(out, "interpretation.csv", caption=caption)
mlca_runtime.record_fact_candidate("top_feature", str(out.iloc[0]["feature"]), source="interpretation.csv")
mlca_runtime.record_metric("n_features_ranked", float(len(importance)), "interpret")
mlca_runtime.record_metric("n_permutation_repeats", 0.0, "interpret")
print(f"train rows={len(train)}; total feature cols={len(feature_cols)}; ranked={len(importance)}")
print(f"method={method}; repeats=0; top={out.iloc[0]['feature']}")
