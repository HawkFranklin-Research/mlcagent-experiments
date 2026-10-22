import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
metric = protocol["task"]["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_key(s):
    try:
        if s.dtype.kind in "OU" and s.nunique() == len(s):
            return True
        if s.dtype.kind in "iu" and s.nunique() == len(s):
            v = s.dropna().sort_values().values
            return len(v) == len(s) and np.array_equal(v, np.arange(v.min(), v.min() + len(v)))
    except Exception:
        pass
    return False

drop = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in drop and not is_key(train[c])]

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inputs["05_preprocess_fit_preprocessing_AER96YQP/preprocessor.pkl"])
sel = load_pkl(inputs["06_features_retain_features_HB34NCSZ/selector_all.pkl"])
base = load_pkl(inputs["07_model_train_baseline_AKKRPP8F/model.pkl"])
alt = load_pkl(inputs["latest_model"])

def final_est(m):
    if hasattr(m, "named_steps") and m.named_steps:
        return list(m.named_steps.values())[-1]
    return m

def params_str(m):
    e = final_est(m)
    p = e.get_params(deep=False) if hasattr(e, "get_params") else {}
    items = []
    for k, v in p.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            items.append(f"{k}={v}")
        else:
            items.append(f"{k}={type(v).__name__}")
    return "; ".join(items[:12])

def top_feature(m, selector, preprocessor):
    names = None
    if hasattr(selector, "feature_names_in_"):
        names = list(selector.feature_names_in_)
    elif hasattr(preprocessor, "feature_names_in_"):
        names = list(preprocessor.feature_names_in_)
    else:
        names = list(feature_cols)
    if hasattr(selector, "get_support"):
        try:
            mask = selector.get_support()
            if len(mask) == len(names):
                names = [n for n, k in zip(names, mask) if k]
        except Exception:
            pass
    e = final_est(m)
    vals = None
    if hasattr(e, "coef_"):
        vals = np.abs(np.asarray(e.coef_))
    elif hasattr(e, "feature_importances_"):
        vals = np.abs(np.asarray(e.feature_importances_))
    elif hasattr(e, "estimators_") and e.estimators_:
        arrs = []
        for sub in e.estimators_:
            if hasattr(sub, "coef_"):
                arrs.append(np.abs(np.asarray(sub.coef_)).ravel())
            elif hasattr(sub, "feature_importances_"):
                arrs.append(np.asarray(sub.feature_importances_).ravel())
        arrs = [a for a in arrs if len(a) == len(names)]
        if arrs:
            vals = np.mean(arrs, axis=0)
    if vals is not None:
        vals = np.asarray(vals)
        if vals.ndim > 1:
            vals = vals.mean(axis=0)
        if len(vals) == len(names):
            return names[int(np.argmax(vals))]
    return "unavailable"

top = top_feature(alt, sel, pre)
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")

v = pd.read_csv(inputs["09_validate_score_validation_TYTHDH5H/validation_metrics.csv"])
headline = None
if "metric" in v.columns and "value" in v.columns:
    sub = v[v["metric"].astype(str).str.lower() == str(metric).lower()]
    if "split" in sub.columns and sub["split"].astype(str).str.lower().eq("val").any():
        sub = sub[sub["split"].astype(str).str.lower() == "val"]
    if "target" in sub.columns and sub["target"].nunique() > 1:
        for t, g in sub.groupby("target"):
            vals = pd.to_numeric(g["value"], errors="coerce").dropna()
            if len(vals):
                mlca_runtime.record_metric(f"val_{metric}_{t}", float(vals.mean()), "val")
    vals = pd.to_numeric(sub["value"], errors="coerce").dropna()
    if len(vals):
        headline = float(vals.mean())
        mlca_runtime.record_metric(f"val_{metric}_mean", headline, "val")

rows = [
    ("targets", ", ".join(targets), np.nan),
    ("primary_metric", str(metric), np.nan),
    ("train_rows", np.nan, float(len(train))),
    ("val_rows", np.nan, float(len(val))),
    ("feature_count", np.nan, float(len(feature_cols))),
    ("baseline_model", type(final_est(base)).__name__, np.nan),
    ("baseline_params", params_str(base), np.nan),
    ("selected_model", type(final_est(alt)).__name__, np.nan),
    ("selected_params", params_str(alt), np.nan),
    ("preprocessor_type", type(pre).__name__, np.nan),
    ("selector_type", type(sel).__name__, np.nan),
    ("validation_metrics_head", v.head(10).to_string(index=False), np.nan),
    ("validation_headline", np.nan, headline if headline is not None else np.nan),
    ("limitations", "Frozen model; no refitting. Validation is the provided val split only. Feature importance is an aggregate over multi-output targets and may not reflect causal effects.", np.nan),
]
interp = pd.DataFrame(rows, columns=["item", "value_str", "value_num"])
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation summary")

print(f"train_rows={len(train)} val_rows={len(val)}")
print(f"features={len(feature_cols)}")
print(f"selected_model={type(final_est(alt)).__name__}")
print(f"top_feature={top}")
print(f"val_{metric}_mean={headline}")
