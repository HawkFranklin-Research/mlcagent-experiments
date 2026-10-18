import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
task = protocol.get("task", {})
targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column", None)
primary_metric = task.get("primary_metric", "silhouette")

def load_pkl(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    is_text = s.dtype.kind in "OU" or pd.api.types.is_string_dtype(s)
    is_int = pd.api.types.is_integer_dtype(s)
    if (is_text or is_int) and s.nunique() == len(s):
        continue
    feature_cols.append(c)

selector = load_pkl("06_features_retain_features_JQHA8ERZ/selector_all.pkl")
model = load_pkl("07_model_train_baseline_7GET20JD/model.pkl")
alt_model = load_pkl("08_model_train_alternative_VX601NWG/alternative_model.pkl")
val_metrics = pd.read_csv(inputs["09_validate_score_validation_ERJRJE6W/validation_metrics.csv"])

def fmt(v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4g}"
    return str(v)

def safe_json(obj):
    def enc(o):
        if isinstance(o, (int, float, str, bool, type(None))):
            return o
        if isinstance(o, dict):
            return {str(k): enc(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [enc(v) for v in o]
        return str(o)
    try:
        return json.dumps(enc(obj))
    except Exception:
        return str(obj)

rows = []
def add(a, v, src):
    rows.append({"aspect": a, "value": fmt(v), "source_file": src})

add("task_type", task.get("type", "clustering"), "_protocol")
add("primary_metric", primary_metric, "_protocol")
add("n_train_rows", len(train), "train")
add("n_val_rows", len(val), "val")
add("n_features", len(feature_cols), "train")
add("feature_columns", ", ".join(feature_cols), "train")
add("baseline_model_class", type(model).__name__, "07_model_train_baseline_7GET20JD/model.pkl")
try:
    add("baseline_hyperparameters", safe_json(model.get_params()), "07_model_train_baseline_7GET20JD/model.pkl")
except Exception:
    add("baseline_hyperparameters", "unavailable", "07_model_train_baseline_7GET20JD/manifest.json")
add("alternative_model_class", type(alt_model).__name__, "08_model_train_alternative_VX601NWG/alternative_model.pkl")
try:
    add("alternative_hyperparameters", safe_json(alt_model.get_params()), "08_model_train_alternative_VX601NWG/alternative_model.pkl")
except Exception:
    add("alternative_hyperparameters", "unavailable", "08_model_train_alternative_VX601NWG/manifest.json")

if len(val_metrics) > 0 and "metric" in val_metrics.columns and "value" in val_metrics.columns:
    for _, r in val_metrics.iterrows():
        add(f"val_{r['metric']}", r["value"], "09_validate_score_validation_ERJRJE6W/validation_metrics.csv")
else:
    for col in val_metrics.columns:
        if pd.api.types.is_numeric_dtype(val_metrics[col]) and len(val_metrics) > 0:
            add(f"val_{col}", float(val_metrics[col].iloc[0]), "09_validate_score_validation_ERJRJE6W/validation_metrics.csv")

sel_names = []
try:
    sup = np.asarray(selector.get_support())
    names = list(getattr(selector, "feature_names_in_", feature_cols))
    if len(names) == len(sup):
        sel_names = [n for n, s in zip(names, sup) if s]
        if sel_names:
            add("selected_features", ", ".join(sel_names), "06_features_retain_features_JQHA8ERZ/selector_all.pkl")
except Exception:
    pass

add("limitations", "clustering has no labeled ground truth; silhouette is internal; no external test split; cluster count from internal criteria; generalization beyond this seed population unclear", "_protocol")
rows = rows[:48]

top_feature = sel_names[0] if sel_names else None
top_source = "06_features_retain_features_JQHA8ERZ/selector_all.pkl" if sel_names else None
if top_feature is None:
    variances = train[feature_cols].var(numeric_only=True)
    if len(variances) > 0:
        top_feature = str(variances.idxmax())
        top_source = "train"

mlca_runtime.save_table(pd.DataFrame(rows), "interpretation.csv", caption="Frozen clustering model interpretation")
if top_feature is not None:
    mlca_runtime.record_fact_candidate("top_feature", top_feature, source=top_source)

print(f"features={len(feature_cols)} train={len(train)} val={len(val)}")
print(f"val_cols={list(val_metrics.columns)[:6]}")
print(f"top_feature={top_feature}")
