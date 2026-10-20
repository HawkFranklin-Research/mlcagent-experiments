import json, pickle, re
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
task = proto["task"]
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
primary = task["primary_metric"]

def load_pkl(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

base = load_pkl("07_model_train_baseline_KWCD46XN/model.pkl")
alt = load_pkl("latest_model")
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
metrics = pd.read_csv(inp["09_validate_score_validation_9MDT3SG5/validation_metrics.csv"])

def steps_of(m):
    if hasattr(m, "named_steps"):
        return list(m.named_steps.keys())
    if hasattr(m, "steps"):
        return [s[0] for s in m.steps]
    return []

def final_est(m):
    if hasattr(m, "named_steps") and m.named_steps:
        return list(m.named_steps.values())[-1]
    if hasattr(m, "steps") and m.steps:
        return m.steps[-1][1]
    return m

def feat_names(m):
    if hasattr(m, "named_steps") and m.named_steps:
        keys = list(m.named_steps.keys())
        for nm in reversed(keys[:-1]):
            try:
                v = list(m.named_steps[nm].get_feature_names_out())
                if v:
                    return v
            except Exception:
                pass
        for obj in [m] + [m.named_steps[k] for k in keys]:
            if hasattr(obj, "feature_names_in_"):
                try:
                    return list(obj.feature_names_in_)
                except Exception:
                    pass
    if hasattr(m, "feature_names_in_"):
        try:
            return list(m.feature_names_in_)
        except Exception:
            pass
    try:
        return list(m.get_feature_names_out())
    except Exception:
        return []

def importance(m):
    est = final_est(m)
    mats = []
    if hasattr(est, "coef_"):
        mats.append(np.abs(np.atleast_2d(est.coef_)).mean(axis=0))
    elif hasattr(est, "feature_importances_"):
        mats.append(np.abs(np.asarray(est.feature_importances_)).ravel())
    if hasattr(est, "estimators_"):
        for e in est.estimators_:
            if hasattr(e, "coef_"):
                mats.append(np.abs(np.atleast_2d(e.coef_)).mean(axis=0))
            elif hasattr(e, "feature_importances_"):
                mats.append(np.abs(np.asarray(e.feature_importances_)).ravel())
    if not mats:
        return None
    n = min(len(x) for x in mats)
    return np.mean([x[:n] for x in mats], axis=0)

def top_feature(m):
    names = feat_names(m)
    im = importance(m)
    if im is not None and len(names) == len(im) and len(im):
        return str(names[int(np.argmax(im))])
    return str(names[0]) if names else "unavailable"

def flat_params(est):
    try:
        out = {k: v for k, v in est.get_params(deep=True).items()
               if isinstance(v, (str, int, float, bool, type(None)))}
    except Exception:
        return []
    pref = [k for k in out if re.search(
        r"n_estimators|max_depth|learning_rate|min_samples|max_iter|solver|penalty|class_weight|C$|alpha|loss|estimator__",
        k, re.I)]
    keys = pref[:12] if pref else list(out)[:8]
    return [(k, out[k]) for k in keys]

def find_metric(df, name):
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            mask = df[c].astype(str).str.contains(name, case=False, regex=False)
            if mask.any():
                row = df[mask].iloc[0]
                for vc in df.columns:
                    if vc == c:
                        continue
                    val = pd.to_numeric(row[vc], errors="coerce")
                    if pd.notna(val):
                        return float(val)
    if name in df.columns:
        s = pd.to_numeric(df[name], errors="coerce").dropna()
        if len(s):
            return float(s.iloc[0])
    return None

def fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)

top = top_feature(alt)
pv = find_metric(metrics, primary)
if pv is not None:
    mlca_runtime.record_metric(primary, pv, "val")

rows = [
    ("model", "baseline_estimator", type(base).__name__, "07_model_train_baseline_KWCD46XN/model.pkl"),
    ("model", "baseline_steps", ",".join(steps_of(base)) or "none", "07_model_train_baseline_KWCD46XN/manifest.json"),
    ("model", "alternative_estimator", type(alt).__name__, "08_model_train_alternative_6JQ0NZNF/alternative_model.pkl"),
    ("model", "alternative_steps", ",".join(steps_of(alt)) or "none", "08_model_train_alternative_6JQ0NZNF/manifest.json"),
    ("input", "train_rows", str(len(train)), "train"),
    ("input", "val_rows", str(len(val)), "val"),
    ("input", "target_count", str(len(targets)), "_protocol"),
    ("input", "excluded_columns", ", ".join(excluded), "_protocol"),
    ("input", "primary_metric", primary, "_protocol"),
    ("validation", "metric_rows", str(len(metrics)), "09_validate_score_validation_9MDT3SG5/validation_metrics.csv"),
]
names = feat_names(alt)
if names:
    rows.append(("input", "feature_count", str(len(names)), "latest_model"))
for k, v in flat_params(final_est(base)):
    rows.append(("hyperparameter", f"baseline.{k}", fmt(v), "07_model_train_baseline_KWCD46XN/manifest.json"))
for k, v in flat_params(final_est(alt)):
    rows.append(("hyperparameter", f"alternative.{k}", fmt(v), "08_model_train_alternative_6JQ0NZNF/manifest.json"))
if pv is not None:
    rows.append(("validation", primary, fmt(pv), "09_validate_score_validation_9MDT3SG5/validation_metrics.csv"))
rows += [
    ("limitations", "refit", "none; interpretation only", "interpretation.csv"),
    ("limitations", "model_scope", "frozen saved pipelines inspected as-is", "interpretation.csv"),
    ("limitations", "validation_scope", "reported validation metrics only; no test exposure", "interpretation.csv"),
    ("top_feature", "selected_feature", top, "latest_model"),
]
interp = pd.DataFrame(rows[:50], columns=["section", "item", "value", "source"])
interp = interp.astype({c: "string" for c in interp.columns})
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation summary")
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")

print("train/val rows:", len(train), len(val))
print("targets:", len(targets), "primary:", primary)
print("top_feature:", top)
if pv is not None:
    print("val", primary, round(pv, 4))
print("saved interpretation.csv")
