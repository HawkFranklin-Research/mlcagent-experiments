import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
avail = list(protocol.keys())
try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    group = protocol["group_column"]
    task = protocol["task"]
except KeyError as e:
    raise KeyError(f"missing protocol key {e}; available: {avail}")
primary = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

cand = [c for c in train.columns if c not in list(targets) + list(excluded) + [group]]
feature_cols = []
for c in cand:
    s = train[c]
    n = int(s.notna().sum())
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        if n > 0 and s.nunique(dropna=True) == n:
            continue
    if pd.api.types.is_integer_dtype(s):
        if n > 0 and s.nunique(dropna=True) == n and s.is_monotonic_increasing:
            continue
    feature_cols.append(c)

model = pickle.load(open(inputs["latest_model"], "rb"))
base = pickle.load(open(inputs["07_model_train_baseline_XYVYQTXX/model.pkl"], "rb"))

def scalar_params(est, prefix=""):
    d = {}
    if hasattr(est, "get_params"):
        for k, v in est.get_params(deep=True).items():
            if isinstance(v, (str, int, float, bool, type(None))):
                d[prefix + k] = v
    return d

params = list(scalar_params(model).items())[:20]

def get_kmeans(est):
    if hasattr(est, "cluster_centers_"):
        return est
    if hasattr(est, "named_steps"):
        for s in est.named_steps.values():
            if hasattr(s, "cluster_centers_"):
                return s
    return None

def transformed_names(est, feature_cols):
    if not hasattr(est, "named_steps"):
        return None
    try:
        return list(est[:-1].get_feature_names_out())
    except Exception:
        pass
    names = list(feature_cols)
    steps = list(est.named_steps.items())
    for nm, st in steps[:-1]:
        if hasattr(st, "get_feature_names_out"):
            try:
                names = list(st.get_feature_names_out(input_features=names))
            except Exception:
                pass
        if hasattr(st, "get_support"):
            try:
                sup = st.get_support()
                names = [n for n, keep in zip(names, sup) if keep]
            except Exception:
                pass
    return names

km = get_kmeans(model)
top_name = "unknown"
top_var = np.nan
if km is not None and hasattr(km, "cluster_centers_"):
    centers = np.asarray(km.cluster_centers_)
    if centers.ndim == 2:
        names = transformed_names(model, feature_cols)
        if names is None or len(names) != centers.shape[1]:
            if hasattr(model, "feature_names_in_") and len(model.feature_names_in_) == centers.shape[1]:
                names = list(model.feature_names_in_)
        if names is not None and len(names) == centers.shape[1]:
            vars_ = np.var(centers, axis=0)
            idx = int(np.argmax(vars_))
            top_name = str(names[idx])
            top_var = float(vars_[idx])

vm = pd.read_csv(inputs["09_validate_score_validation_TZGPKJKF/validation_metrics.csv"])
val_metric = np.nan
if "metric" in vm.columns and "value" in vm.columns:
    m = vm[vm["metric"].astype(str).str.lower() == str(primary).lower()]
    if not m.empty:
        try:
            val_metric = float(m["value"].iloc[0])
        except Exception:
            pass
if np.isnan(val_metric):
    for col in vm.columns:
        if "value" in col.lower() or "score" in col.lower():
            try:
                val_metric = float(vm[col].iloc[0])
                break
            except Exception:
                pass

def rs(x):
    try:
        x = float(x)
        if x == 0:
            return 0.0
        return float(f"{x:.4g}")
    except Exception:
        return x

rows = [
    {"section": "model", "item": "class", "value_str": type(model).__name__, "value_num": np.nan},
    {"section": "model", "item": "baseline_class", "value_str": type(base).__name__, "value_num": np.nan},
]
for k, v in params:
    rows.append({
        "section": "hyperparameter",
        "item": k,
        "value_str": str(v),
        "value_num": rs(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan,
    })
rows.extend([
    {"section": "features", "item": "n_features", "value_str": np.nan, "value_num": rs(len(feature_cols))},
    {"section": "features", "item": "top_feature", "value_str": top_name, "value_num": rs(top_var)},
    {"section": "validation", "item": primary, "value_str": np.nan, "value_num": rs(val_metric)},
    {"section": "limitations", "item": "no_outcome", "value_str": "Clustering has no held-out labels; silhouette is an internal criterion.", "value_num": np.nan},
    {"section": "limitations", "item": "customer_aggregation", "value_str": "Rows are invoice lines; the brief describes per-customer segments while this frozen model consumes row-level features.", "value_num": np.nan},
])

df = pd.DataFrame(rows, columns=["section", "item", "value_str", "value_num"])
mlca_runtime.save_table(df, "interpretation.csv", caption="Frozen model interpretation")
mlca_runtime.record_fact_candidate("top_feature", top_name, source="interpretation.csv")
if not np.isnan(val_metric):
    mlca_runtime.record_metric(primary, float(val_metric), "val")

print(f"train rows: {len(train)}")
print(f"val rows: {len(val)}")
print(f"features: {len(feature_cols)}")
print(f"model: {type(model).__name__}")
print(f"{primary}: {val_metric}, top_feature: {top_name}")
