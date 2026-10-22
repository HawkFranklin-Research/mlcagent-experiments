import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

_protocol = json.loads(inputs["_protocol"].read_text())
for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in _protocol:
        raise KeyError(f"Missing protocol key {k!r}; available: {list(_protocol.keys())}")
targets = list(_protocol["targets"])
excluded = list(_protocol["excluded_columns"])
group_col = _protocol["group_column"]
task = _protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; task keys: {list(task.keys())}")
primary_metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_keylike(s):
    if not s.is_unique:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        name = str(s.name).lower()
        return any(t in name for t in ("id", "index", "row", "code"))
    return False

reserved = set(targets + excluded + [group_col])
feature_cols = [c for c in train.columns if c not in reserved and not is_keylike(train[c])]

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def final_estimator(m):
    if hasattr(m, "steps"):
        return m.steps[-1][1]
    return m

def get_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "named_steps"):
        for _, step in reversed(list(obj.named_steps.items())):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

model = load_pkl(inputs["latest_model"])
fe = final_estimator(model)

imp = None
if hasattr(fe, "coef_"):
    coef = np.asarray(fe.coef_)
    imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
elif hasattr(fe, "feature_importances_"):
    imp = np.asarray(fe.feature_importances_)

names = get_names(fe) or get_names(model)
if names is None:
    sel0 = load_pkl(inputs["06_features_retain_features_KKEEW75S/selector_all.pkl"])
    names = get_names(sel0)

top_feature = "unknown"
if imp is not None and names is not None and len(names) == len(imp):
    top_feature = str(names[int(np.argmax(imp))])
elif imp is not None and names is not None:
    sel = load_pkl(inputs["06_features_retain_features_KKEEW75S/selector_all.pkl"])
    if hasattr(sel, "get_support"):
        try:
            mask = sel.get_support()
            if len(mask) == len(names):
                selected = np.asarray(names)[mask]
                if len(selected) == len(imp):
                    top_feature = str(selected[int(np.argmax(imp))])
        except Exception:
            pass

manifest_path = inputs["08_model_train_alternative_A6RD3R24/manifest.json"]
try:
    manifest = json.loads(manifest_path.read_text())
except Exception:
    manifest = {}
params = manifest.get("params") or manifest.get("hyperparameters") or manifest.get("model_params") or {}
if not params and hasattr(fe, "get_params"):
    try:
        params = {k: v for k, v in fe.get_params(deep=False).items()
                  if isinstance(v, (str, int, float, bool, type(None)))}
    except Exception:
        params = {}

vm = pd.read_csv(inputs["09_validate_score_validation_Z3ATC4H1/validation_metrics.csv"])
metric_value = np.nan
if primary_metric in vm.columns and len(vm):
    metric_value = vm[primary_metric].iloc[0]
elif {"metric", "value"}.issubset(vm.columns):
    hit = vm[vm["metric"] == primary_metric]
    if not hit.empty:
        metric_value = hit["value"].iloc[0]
else:
    nums = vm.select_dtypes(include=[np.number])
    if not nums.empty:
        metric_value = nums.iloc[0, 0]

rows = [
    ("task", "primary_metric", primary_metric, "_protocol"),
    ("task", "targets", json.dumps(targets), "_protocol"),
    ("data", "group_column", group_col, "_protocol"),
    ("data", "excluded_columns", json.dumps(excluded), "_protocol"),
    ("data", "train_rows", str(len(train)), "train.parquet"),
    ("data", "val_rows", str(len(val)), "val.parquet"),
    ("features", "n_features", str(len(feature_cols)), "train.parquet"),
    ("features", "feature_columns", ", ".join(map(str, feature_cols)), "train.parquet"),
    ("model", "class", type(fe).__name__, "08_model_train_alternative_A6RD3R24/alternative_model.pkl"),
    ("model", "hyperparameters", json.dumps(params, default=str)[:1000], "08_model_train_alternative_A6RD3R24/manifest.json"),
    ("validation", primary_metric, str(metric_value), "09_validate_score_validation_Z3ATC4H1/validation_metrics.csv"),
    ("interpretation", "top_feature", top_feature, "08_model_train_alternative_A6RD3R24/alternative_model.pkl"),
    ("limitations", "note", "No fitting performed; evidence comes from frozen artifacts only.", "interpretation.csv"),
]
interp = pd.DataFrame(rows, columns=["section", "item", "value", "source_file"])
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation summary")

if top_feature != "unknown":
    mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print("train rows:", len(train), "val rows:", len(val))
print("features:", len(feature_cols))
print("primary metric:", primary_metric, "value:", metric_value)
print("top feature:", top_feature)
print("saved interpretation.csv")
