import mlca_runtime, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

def read_json(p):
    return json.loads(Path(p).read_text())

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

proto = read_json(inp["_protocol"])

def pget(path):
    cur = proto
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing {k!r} at {path}; available top-level keys: {list(proto.keys())}")
        cur = cur[k]
    return cur

targets = list(pget(["targets"]))
excluded = list(pget(["excluded_columns"]))
group = pget(["group_column"])
primary_metric = pget(["task", "primary_metric"])

model_key = next((k for k in ["latest_model", "07_model_train_baseline_HK97MR3S/model.pkl",
                              "08_model_train_alternative_2D5YHWB5/alternative_model.pkl"]
                  if k in inp and Path(inp[k]).exists()), None)
if model_key is None:
    raise KeyError("No frozen model input found")
model = load_pkl(inp[model_key])

def final_est(est):
    return list(est.named_steps.values())[-1] if hasattr(est, "named_steps") else est

fin = final_est(model)
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def is_keylike(s, name):
    n, nu = len(s), s.nunique(dropna=True)
    if n <= 50 or nu != n:
        return False
    if isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s.dtype):
        return True
    return pd.api.types.is_integer_dtype(s.dtype) and any(t in str(name).lower() for t in ("id","index","code","key"))

def feature_cols(df):
    g = {group} if isinstance(group, str) else set(group or [])
    drop = set(targets) | set(excluded) | g
    return [c for c in df.columns if c not in drop and not is_keylike(df[c], c)]

raw_feats = feature_cols(train)

model_nm = None
for obj in (model, fin):
    if hasattr(obj, "feature_names_in_"):
        model_nm = [str(x) for x in np.asarray(obj.feature_names_in_).ravel()]
        break
if model_nm is None:
    model_nm = raw_feats

vm = pd.read_csv(inp["09_validate_score_validation_C1P5YKW9/validation_metrics.csv"])
recs = []
if len(vm) == 1:
    for c in vm.columns:
        try:
            recs.append((str(c), float(vm.iloc[0][c])))
        except Exception:
            pass
prim_val = next((v for n, v in recs if primary_metric.lower() in n.lower()), np.nan)
if not np.isfinite(prim_val) and len(recs) == 1:
    prim_val = recs[0][1]
if np.isfinite(prim_val):
    mlca_runtime.record_metric(f"val_{primary_metric}", float(prim_val), "val")

def sfmt(v):
    if v is None:
        return ""
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return "" if not np.isfinite(v) else f"{v:.4g}"
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if isinstance(v, (list, tuple)):
        return "; ".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, default=str)
    return str(v)

rows = []
def add(section, item, value, source=""):
    rows.append({"section": str(section), "item": str(item), "value": sfmt(value), "source": str(source)})

msrc = Path(inp[model_key]).name
add("model", "frozen_model_file", msrc, msrc)
add("model", "model_class", type(model).__name__, msrc)
if hasattr(model, "named_steps"):
    add("model", "pipeline_steps", list(model.named_steps.keys()), msrc)
add("model", "final_estimator", type(fin).__name__, msrc)
params = fin.get_params(deep=False) if hasattr(fin, "get_params") else {}
for k in ["C", "solver", "max_iter", "class_weight", "penalty", "n_estimators", "max_depth", "learning_rate"]:
    if k in params:
        add("hyperparameters", k, params[k], msrc)
inner = getattr(fin, "estimator", None)
if inner is not None and hasattr(inner, "get_params"):
    ip = inner.get_params(deep=False)
    for k in ["C", "solver", "max_iter", "class_weight", "penalty"]:
        if k in ip:
            add("hyperparameters", f"inner_{k}", ip[k], msrc)

add("inputs", "targets", targets, Path(inp["_protocol"]).name)
add("inputs", "raw_feature_count", len(raw_feats), "train.parquet")
add("inputs", "model_input_feature_count", len(model_nm), msrc)
add("inputs", "raw_features_preview", raw_feats[:20], "train.parquet")

add("validation", "primary_metric", primary_metric, Path(inp["_protocol"]).name)
add("validation", f"val_{primary_metric}", prim_val, Path(inp["09_validate_score_validation_C1P5YKW9/validation_metrics.csv"]).name)
add("validation", "validation_metrics_rows", len(vm), Path(inp["09_validate_score_validation_C1P5YKW9/validation_metrics.csv"]).name)

add("limitations", "no_refit", "Frozen model interpreted without fitting.", "")
add("limitations", "importance_method", "Model coefficients/feature_importances_ only; no permutation or refit.", msrc)
add("limitations", "validation_scope", "Validation evidence is read-only; test data not used.", "")
add("limitations", "multilabel_targets", "Seven binary fault targets treated as joint multilabel outcomes.", Path(inp["_protocol"]).name)

interp = pd.DataFrame(rows).astype(str)
mlca_runtime.save_table(interp, "interpretation.csv")

print(f"train={train.shape} val={val.shape}")
print(f"model={type(model).__name__} file={msrc}")
print(f"primary_metric={primary_metric} val={prim_val}")
print(f"saved interpretation.csv rows={len(interp)}")
