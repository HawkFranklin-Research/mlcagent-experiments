import mlca_runtime, json, pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")
task = protocol["task"]
metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def lp(k):
    with open(inputs[k], "rb") as f:
        return pickle.load(f)

model = lp("07_model_train_baseline_25GK6Z71/model.pkl")
alt = lp("latest_model")
sel = lp("06_features_retain_features_XC2ENAE2/selector_all.pkl")
man_b = json.loads(inputs["07_model_train_baseline_25GK6Z71/manifest.json"].read_text())
man_a = json.loads(inputs["08_model_train_alternative_VF3B36TN/manifest.json"].read_text())
vm = pd.read_csv(inputs["09_validate_score_validation_8TJRCB1S/validation_metrics.csv"])

feat_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]

def unwrap_attrs(s):
    yield s
    for a in ("regressor_", "estimator_", "best_estimator_"):
        if hasattr(s, a):
            yield getattr(s, a)

def describe(m):
    steps = list(m.named_steps.values()) if isinstance(m, Pipeline) else [m]
    last = steps[-1]
    c, imp = None, None
    for cand in unwrap_attrs(last):
        if hasattr(cand, "feature_importances_"):
            c, imp = cand, np.asarray(cand.feature_importances_).ravel(); break
        if hasattr(cand, "coef_"):
            a = np.asarray(cand.coef_)
            if a.ndim > 1: a = np.abs(a).mean(axis=0)
            c, imp = cand, np.abs(np.ravel(a)); break
    return type(last).__name__, {k: (v if isinstance(v,(int,float,str,bool)) or v is None else type(v).__name__) for k,v in last.get_params().items()}, c, imp

def top_feature(m):
    _, _, cand, imp = describe(m)
    if cand is None or imp is None or imp.size == 0:
        return None
    names = getattr(cand, "feature_names_in_", None)
    if names is not None and len(names) == imp.size:
        return str(names[int(np.argmax(imp))])
    sels = []
    steps = list(m.named_steps.values()) if isinstance(m, Pipeline) else [m]
    for s in steps:
        if hasattr(s, "get_support") and hasattr(s, "feature_names_in_"):
            sels.append(s)
    fr = list(sel.named_steps.values())[0] if isinstance(sel, Pipeline) else sel
    if hasattr(fr, "get_support") and hasattr(fr, "feature_names_in_"):
        sels.append(fr)
    for s in sels:
        support = np.asarray(s.get_support())
        nm = np.asarray(s.feature_names_in_)[support]
        if len(nm) == imp.size:
            return str(nm[int(np.argmax(imp))])
    return None

alt_name, alt_params, _, _ = describe(alt)
base_name, base_params, _, _ = describe(model)
top = top_feature(alt) or top_feature(model)

val_score = None
for col in vm.columns:
    if col.lower().strip() == metric.lower():
        try: val_score = float(vm[col].iloc[0])
        except Exception: pass
        break

rows = []
def add(sec, item, val, src):
    rows.append({"section": sec, "item": item, "value": val, "source": src})

add("model", "alternative_estimator", alt_name, "08_model_train_alternative_VF3B36TN/alternative_model.pkl")
add("model", "baseline_estimator", base_name, "07_model_train_baseline_25GK6Z71/model.pkl")
add("model", "alternative_hyperparameters", json.dumps(alt_params), "08_model_train_alternative_VF3B36TN/manifest.json")
add("model", "baseline_hyperparameters", json.dumps(base_params), "07_model_train_baseline_25GK6Z71/manifest.json")
add("inputs", "targets", json.dumps(targets), "_protocol")
add("inputs", "feature_count", str(len(feat_cols)), "train.parquet")
add("inputs", "feature_columns", json.dumps(feat_cols), "train.parquet")
add("inputs", "train_rows", str(len(train)), "train.parquet")
add("inputs", "val_rows", str(len(val)), "val.parquet")
add("validation", "primary_metric", metric, "_protocol")
for col in vm.columns:
    try: add("validation", str(col), str(vm[col].iloc[0]), "09_validate_score_validation_8TJRCB1S/validation_metrics.csv")
    except Exception: pass
if top: add("model", "top_feature", top, "alternative_model.pkl")
add("limitations", "no_refit", "Interpretation only; frozen models read, nothing refit or scored.", "n/a")
add("limitations", "single_target", "Analysis limited to declared target(s): " + json.dumps(targets), "_protocol")

interp = pd.DataFrame(rows)
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation summary")

if val_score is not None:
    mlca_runtime.record_metric(metric, val_score, "val")
if top:
    mlca_runtime.record_fact_candidate("top_feature", top, source="08_model_train_alternative_VF3B36TN/alternative_model.pkl")

print("features:", len(feat_cols), "train:", len(train), "val:", len(val), "metric:", val_score)
