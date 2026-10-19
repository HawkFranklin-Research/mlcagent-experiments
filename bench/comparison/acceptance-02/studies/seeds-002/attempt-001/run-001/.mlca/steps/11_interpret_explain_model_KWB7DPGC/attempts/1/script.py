import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(Path(inputs["_protocol"]).read_text())
for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in proto:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(proto)}")

task = proto["task"]
metric_name = task.get("primary_metric") or "silhouette"
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group = proto["group_column"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
drop = set(targets) | set(excluded) | ({group} if group is not None else set())
feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if s.is_unique and (not pd.api.types.is_numeric_dtype(s.dtype) or pd.api.types.is_integer_dtype(s.dtype)):
        continue
    feature_cols.append(c)


def simple_params(est):
    out = []
    try:
        params = est.get_params(deep=False)
    except Exception:
        return "unavailable"
    for k, v in params.items():
        if isinstance(v, bool) or v is None or isinstance(v, (str, int)):
            out.append(f"{k}={v}")
        elif isinstance(v, float):
            out.append(f"{k}={v:.4g}")
        elif isinstance(v, (list, tuple)) and len(v) <= 5 and all(
            isinstance(x, (str, int, float, bool, type(None))) for x in v
        ):
            out.append(f"{k}={v}")
    return "; ".join(out) if out else "unavailable"


with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)
named = getattr(model, "named_steps", None)
if named:
    est = list(named.values())[-1]
    model_type = f"{type(model).__name__}({type(est).__name__})"
    hyper = simple_params(est)
else:
    est = model
    model_type = type(model).__name__
    hyper = simple_params(model)

top_feature = "unknown"
top_source = "07_model_train_baseline_SAV13199/train_clusters.parquet"
try:
    tc = pd.read_parquet(inputs["07_model_train_baseline_SAV13199/train_clusters.parquet"])
    if "row_id" in tc.columns:
        tc = tc.set_index("row_id")
    ccol = "cluster" if "cluster" in tc.columns else [c for c in tc.columns if c != "row_id"][0]
    joined = train.loc[tc.index, feature_cols].join(tc[[ccol]], how="inner")
    best = (-1.0, None)
    for c in feature_cols:
        x = pd.to_numeric(joined[c], errors="coerce")
        mask = x.notna()
        if mask.sum() == 0:
            continue
        g = joined.loc[mask, ccol]
        x = x[mask]
        overall = x.mean()
        total = ((x - overall) ** 2).sum()
        if total <= 0:
            continue
        between = sum(len(x[g == k]) * ((x[g == k].mean() - overall) ** 2) for k in g.unique())
        ratio = between / total
        if ratio > best[0]:
            best = (ratio, c)
    if best[1] is not None:
        top_feature = best[1]
except Exception:
    pass

val_value = None
try:
    vm = pd.read_csv(inputs["09_validate_score_validation_AE76P4WA/validation_metrics.csv"])
    for col in vm.columns:
        if metric_name.lower() in str(col).lower():
            s = pd.to_numeric(vm[col], errors="coerce").dropna()
            if len(s):
                val_value = float(s.iloc[0])
                break
    if val_value is None:
        for col in vm.select_dtypes(include="object").columns:
            mask = vm[col].astype(str).str.contains(metric_name, case=False, na=False)
            if mask.any():
                row = vm[mask].iloc[0]
                nums = pd.to_numeric(row, errors="coerce").dropna()
                if len(nums):
                    val_value = float(nums.iloc[0])
                    break
    if val_value is None:
        nums = vm.select_dtypes(include="number")
        if nums.shape[1]:
            val_value = float(nums.iloc[0, 0])
except Exception:
    pass
if val_value is not None:
    mlca_runtime.record_metric(metric_name, val_value, "val")

rows = []
def add(section, detail, source):
    rows.append({"section": section, "detail": str(detail), "source_file": source})

val_detail = f"{val_value:.4g}" if val_value is not None else "not found"
add("model_type", model_type, "latest_model")
add("selected_hyperparameters", hyper, "latest_model")
add("input_features", ", ".join(feature_cols), "train.parquet")
add("n_features", len(feature_cols), "train.parquet")
add("n_train_rows", len(train), "train.parquet")
add("n_validation_rows", len(val), "val.parquet")
add("primary_metric", metric_name, "_protocol")
add("validation_value", val_detail, "09_validate_score_validation_AE76P4WA/validation_metrics.csv")
add("limitation", "No outcome or label column; clustering quality is internal only, not predictive accuracy.", "_protocol")
add("limitation", "Geometry-only features; no external or process variables.", "train.parquet")
add("limitation", "Cluster count is chosen from internal criteria and may not match botanical groupings.", "latest_model")
add("limitation", "Validation evidence reflects the frozen model on the validation split only; no test exposure.", "09_validate_score_validation_AE76P4WA/validation_metrics.csv")

interp = pd.DataFrame(rows).head(50)
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen clustering model interpretation")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source=top_source)
print("interpretation.csv saved")
print(f"features={len(feature_cols)} train={len(train)} val={len(val)}")
print(f"top_feature={top_feature}")
if val_value is not None:
    print(f"{metric_name}={val_value:.4g}")
