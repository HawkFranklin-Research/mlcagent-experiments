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
feature_cols = [c for c in train.columns
                if c not in excluded_set and not is_keylike(train[c])]

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

def find_source(est, depth=0):
    if depth > 5:
        return None
    if hasattr(est, "coef_") or hasattr(est, "feature_importances_"):
        return est
    children = []
    for _, s in reversed(getattr(est, "steps", [])):
        children.append(s)
    for attr in ("estimators_", "calibrated_classifiers_", "named_estimators_"):
        v = getattr(est, attr, None)
        if v is not None:
            children.extend(v.values() if isinstance(v, dict) else list(v))
    for attr in ("estimator", "final_estimator_", "best_estimator_"):
        v = getattr(est, attr, None)
        if v is not None:
            children.append(v)
    for child in children:
        if isinstance(child, tuple) and len(child) == 2:
            child = child[1]
        r = find_source(child, depth + 1)
        if r is not None:
            return r
    return None

src = find_source(model)
if src is None:
    raise RuntimeError("frozen model tree has no coef_ or feature_importances_")

if hasattr(src, "coef_"):
    c = np.asarray(src.coef_)
    if c.ndim > 1:
        imp = np.abs(c).mean(0); sd = np.abs(c).std(0)
    else:
        imp = np.abs(c); sd = np.zeros_like(imp)
    kind = "coef"
else:
    imp = np.asarray(src.feature_importances_); sd = np.zeros_like(imp)
    kind = "fi"

raw_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else feature_cols
names = None
if len(raw_features) == len(imp):
    names = list(raw_features)
if names is None and hasattr(model, "steps"):
    for _, s in reversed(model.steps[:-1]):
        if hasattr(s, "get_feature_names_out"):
            try:
                g = list(s.get_feature_names_out())
                if len(g) == len(imp):
                    names = g; break
            except Exception:
                pass
if names is None and hasattr(model, "get_feature_names_out"):
    try:
        g = list(model.get_feature_names_out())
        if len(g) == len(imp): names = g
    except Exception:
        pass

if names is None or len(names) != len(imp):
    nout = min(15, len(imp))
    order = np.argsort(-imp)[:nout]
    res = pd.DataFrame({
        "feature": [f"feature_{int(i)}" for i in order],
        "importance_mean": np.round(imp[order], 4),
        "importance_sd": np.round(sd[order], 4),
    })
    caption = f"method={kind}_nested; limitation=cannot_map_importances_to_raw_feature_names; ranked={len(imp)}"
    mlca_runtime.save_table(res, "interpretation.csv", caption=caption)
    mlca_runtime.record_fact_candidate("top_feature", res.iloc[0]["feature"], source="interpretation.csv")
    print(f"limitation; ranked={len(imp)}; top={res.iloc[0]['feature']}")
else:
    order = np.argsort(-imp)[:15]
    res = pd.DataFrame({
        "feature": [names[i] for i in order],
        "importance_mean": np.round(imp[order], 4),
        "importance_sd": np.round(sd[order], 4),
    })
    caption = f"method={kind}_nested; frozen_model=latest_model; ranked={len(imp)}; top15_shown"
    mlca_runtime.save_table(res, "interpretation.csv", caption=caption)
    mlca_runtime.record_fact_candidate("top_feature", res.iloc[0]["feature"], source="interpretation.csv")
    print(f"method={kind}; ranked={len(imp)}; top={res.iloc[0]['feature']}")
