import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()


def req(d, k, ctx):
    if k not in d:
        raise KeyError(f"Missing {k!r} in {ctx}; available keys: {sorted(d)}")
    return d[k]


protocol = json.loads(inp["_protocol"].read_text())
targets = req(protocol, "targets", "_protocol")
excluded = req(protocol, "excluded_columns", "_protocol")
group_col = req(protocol, "group_column", "_protocol")
task = req(protocol, "task", "_protocol")
primary_metric = req(task, "primary_metric", "_protocol.task") or "silhouette"


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def feature_names(obj):
    n = getattr(obj, "feature_names_in_", None)
    if n is not None:
        return [str(x) for x in list(n)]
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            n = getattr(step, "feature_names_in_", None)
            if n is not None:
                return [str(x) for x in list(n)]
    return []


def steps_str(obj):
    if isinstance(obj, Pipeline):
        return " -> ".join(str(k) for k in obj.named_steps.keys())
    return type(obj).__name__


def final_est(obj):
    if isinstance(obj, Pipeline):
        return list(obj.named_steps.values())[-1]
    return obj


def params_str(est):
    if not hasattr(est, "get_params"):
        return ""
    try:
        params = est.get_params(deep=False)
    except Exception:
        return ""
    parts = []
    for k, v in params.items():
        if k in ("memory", "steps", "verbose"):
            continue
        s = repr(v)
        if len(s) > 80:
            s = s[:77] + "..."
        parts.append(f"{k}={s}")
    return "; ".join(parts[:12])


model = load_pkl(inp["latest_model"])
final = final_est(model)
fnames = feature_names(model)
if not fnames:
    selector = load_pkl(inp["06_features_retain_features_5RGESHAY/selector_all.pkl"])
    fnames = feature_names(selector)

top_feature = "not_applicable_no_supervised_ranking"
imp = getattr(final, "feature_importances_", None)
if imp is not None and len(fnames) == len(imp):
    top_feature = fnames[int(pd.Series(imp).abs().idxmax())]

val_df = pd.read_csv(inp["09_validate_score_validation_ERHSD5DA/validation_metrics.csv"])
metric_rows = []
for c in val_df.columns:
    if pd.api.types.is_numeric_dtype(val_df[c]):
        s = pd.to_numeric(val_df[c], errors="coerce").dropna()
        if len(s):
            value = float(s.iloc[-1] if len(s) == 1 else s.mean())
            metric_rows.append((c, value))

for c, v in metric_rows:
    if "silhouette" in c.lower():
        mlca_runtime.record_metric("silhouette", v, "val")
        break

rows = []


def add(aspect, detail, source):
    rows.append({"aspect": aspect, "detail": str(detail), "source": source})


add("task_type", task.get("type", ""), "_protocol")
add("primary_metric", primary_metric, "_protocol")
add("targets", json.dumps(targets), "_protocol")
add("excluded_columns", json.dumps(excluded), "_protocol")
add("group_column", group_col, "_protocol")
add("model_type", type(final).__name__, "latest_model")
add("model_steps", steps_str(model), "latest_model")
add("selected_hyperparameters", params_str(final), "latest_model")
add("input_feature_count", len(fnames) if fnames else "unknown", "latest_model")
add("input_features", ", ".join(fnames[:20]) + ("..." if len(fnames) > 20 else ""), "latest_model")
add("top_feature_candidate", top_feature, "interpretation.csv")

for c, v in metric_rows[:8]:
    add(f"validation_{c}", f"{v:.4g}", "09_validate_score_validation_ERHSD5DA/validation_metrics.csv")
if not metric_rows:
    add("validation_numeric_metrics", "none_found", "09_validate_score_validation_ERHSD5DA/validation_metrics.csv")

add("limitation_no_supervised_outcome", "No target; clustering is validated only with internal metrics.", "interpretation.csv")
add("limitation_validation_only", "Validation evidence is reported; no post-validation test refit is performed.", "interpretation.csv")
add("limitation_frozen", "Frozen pipeline is described without refitting on train or validation.", "interpretation.csv")
add("limitation_grouping", f"Rows are grouped by {group_col}; entity-level aggregation must be preserved.", "interpretation.csv")

interp = pd.DataFrame(rows)
interp_path = mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source=interp_path.name)

print("interpretation rows:", len(interp))
print("model:", type(final).__name__, "steps:", steps_str(model))
print("features:", len(fnames), "primary_metric:", primary_metric)
print("validation metric columns:", len(metric_rows))
