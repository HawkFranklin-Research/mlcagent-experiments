import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

protocol = json.loads(inp["_protocol"].read_text())
def req(d, k):
    if k not in d:
        raise KeyError(f"Missing protocol key {k!r}; available top-level keys: {list(d.keys())}")
    return d[k]
task = req(protocol, "task")
primary = req(task, "primary_metric")
targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def load_pkl(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("05_preprocess_fit_preprocessing_KQAA3HGR/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_SPWDSE6R/selector_all.pkl")
base = load_pkl("07_model_train_baseline_726ZEXFH/model.pkl")
alt = load_pkl("08_model_train_alternative_R61594N7/alternative_model.pkl")
latest = load_pkl("latest_model")

def final_est(model):
    obj = model
    while hasattr(obj, "named_steps"):
        steps = list(obj.named_steps.values())
        if not steps:
            break
        obj = steps[-1]
    return obj

def steps_text(model):
    if hasattr(model, "named_steps"):
        return " -> ".join(f"{k}:{type(v).__name__}" for k, v in model.named_steps.items())
    return type(model).__name__

def params_text(est):
    try:
        p = est.get_params(deep=False)
    except Exception:
        return "{}"
    clean = {}
    for k, v in p.items():
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating, float)):
            v = float(f"{float(v):.4g}")
        elif not isinstance(v, (str, int, float, bool, type(None))):
            v = str(v)
        clean[k] = v
    return json.dumps(clean, sort_keys=True, default=str)

def input_features(model):
    for obj in [model, pre, sel]:
        if hasattr(obj, "feature_names_in_"):
            return list(np.asarray(obj.feature_names_in_).astype(str))
    return []

latest_est = final_est(latest)
base_est = final_est(base)

def importance_vector(est):
    for attr in ["coef_", "feature_importances_"]:
        if hasattr(est, attr):
            arr = np.abs(np.asarray(getattr(est, attr), dtype=float))
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            return arr.ravel()
    if hasattr(est, "cluster_centers_"):
        centers = np.asarray(est.cluster_centers_, dtype=float)
        return centers.var(axis=0).ravel()
    return None

def names_for_est(model, est, n):
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.items())
        try:
            names = model[:-1].get_feature_names_out()
            if len(names) == n:
                return [str(x) for x in names]
        except Exception:
            pass
        for _, obj in steps[:-1]:
            try:
                names = obj.get_feature_names_out()
                if len(names) == n:
                    return [str(x) for x in names]
            except Exception:
                pass
    for obj in [sel, pre]:
        try:
            names = obj.get_feature_names_out()
            if len(names) == n:
                return [str(x) for x in names]
        except Exception:
            pass
    for obj in [model, est, sel, pre]:
        if hasattr(obj, "feature_names_in_"):
            names = list(np.asarray(obj.feature_names_in_).astype(str))
            if len(names) == n:
                return names
    return [f"feature_{i}" for i in range(n)]

def clean_name(name):
    s = str(name)
    return s.split("__", 1)[1] if "__" in s else s

imp = importance_vector(latest_est)
if imp is None or imp.size == 0:
    top_feature = "not_available"
else:
    names = names_for_est(latest, latest_est, len(imp))
    top_feature = clean_name(names[int(np.argmax(imp))])

vm = pd.read_csv(inp["09_validate_score_validation_3PNV70HF/validation_metrics.csv"])
def find_metric(df, name):
    nl = str(name).lower()
    for c in df.columns:
        if str(c).strip().lower() == nl:
            try: return float(df.iloc[0][c])
            except Exception: return np.nan
    for mcol in ["metric", "metric_name", "name", "measure"]:
        if mcol in df.columns:
            vals = df[mcol].astype(str).str.lower()
            mask = vals.eq(nl)
            if not mask.any():
                mask = vals.str.contains(nl, regex=False, na=False)
            if mask.any():
                for vcol in ["value", "score", "mean", "val", "estimate"]:
                    if vcol in df.columns:
                        try: return float(df.loc[mask, vcol].iloc[0])
                        except Exception: return np.nan
                nums = df.loc[mask].select_dtypes(include=[np.number])
                if nums.shape[1]:
                    return float(nums.iloc[0, 0])
    for c in df.columns:
        if nl in str(c).lower():
            try: return float(df.iloc[0][c])
            except Exception: return np.nan
    nums = df.select_dtypes(include=[np.number])
    return float(nums.iloc[0, 0]) if nums.size == 1 else np.nan

val_value = find_metric(vm, primary)
raw_features = input_features(latest)
feature_text = ", ".join(raw_features[:20])
if len(raw_features) > 20:
    feature_text += f" ... (+{len(raw_features) - 20})"
limitations = (
    "No labeled outcome; clustering evidence is internal and exploratory. "
    "Silhouette is not a measure of business validity. Rows are transactional line items, "
    "not independent customers; clusters are not customer segments unless aggregation was applied. "
    "No temporal holdout is declared, so stability over time is not established."
)

rows = []
def add(item, text=None, num=None, source=""):
    rows.append({
        "item": item,
        "value_text": np.nan if text is None else str(text),
        "value_numeric": np.nan if num is None else float(num),
        "source": source,
    })

add("primary_metric", primary, None, inp["_protocol"].name)
add("train_rows", None, len(train), inp["train"].name)
add("val_rows", None, len(val), inp["val"].name)
add("validation_value", None, val_value, inp["09_validate_score_validation_3PNV70HF/validation_metrics.csv"].name)
add("latest_model_steps", steps_text(latest), None, inp["latest_model"].name)
add("baseline_model_steps", steps_text(base), None, inp["07_model_train_baseline_726ZEXFH/model.pkl"].name)
add("latest_hyperparameters", params_text(latest_est), None, inp["latest_model"].name)
add("baseline_hyperparameters", params_text(base_est), None, inp["07_model_train_baseline_726ZEXFH/model.pkl"].name)
add("raw_input_feature_count", None, len(raw_features), inp["latest_model"].name)
add("raw_input_features", feature_text or "not_available", None, inp["latest_model"].name)
add("top_feature", top_feature, None, "interpretation.csv")
add("limitations", limitations, None, inp["_protocol"].name)

df = pd.DataFrame(rows)
df["value_numeric"] = df["value_numeric"].map(
    lambda x: np.nan if pd.isna(x) else float(f"{float(x):.4g}")
)
mlca_runtime.save_table(df, "interpretation.csv", caption="Frozen model interpretation")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"train rows={len(train)}, val rows={len(val)}")
print(f"primary_metric={primary}, validation_value={val_value}")
print(f"top_feature={top_feature}")
