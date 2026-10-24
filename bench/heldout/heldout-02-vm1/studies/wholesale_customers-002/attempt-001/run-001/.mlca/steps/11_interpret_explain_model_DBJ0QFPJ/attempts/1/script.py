import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

def read_json(key):
    return json.loads(inp[key].read_text())

def load_pkl(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

protocol = read_json("_protocol")
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {list(protocol.keys())}")
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; task keys: {list(task.keys())}")
primary = task["primary_metric"]
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
print("train_rows", len(train), "val_rows", len(val))

pre = load_pkl("05_preprocess_fit_preprocessing_WG30AQ6C/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_SSZF8P1C/selector_all.pkl")
base = load_pkl("07_model_train_baseline_PYZ8RSVK/model.pkl")
latest = load_pkl("latest_model")
base_manifest = read_json("07_model_train_baseline_PYZ8RSVK/manifest.json")
alt_manifest = read_json("08_model_train_alternative_3CGWKKCJ/manifest.json")
val_manifest = read_json("09_validate_score_validation_RBGGTR3K/manifest.json")
metrics = pd.read_csv(inp["09_validate_score_validation_RBGGTR3K/validation_metrics.csv"])

def final_est(obj):
    return obj.steps[-1][1] if isinstance(obj, Pipeline) else obj

def find_metric(df, name):
    n = name.lower()
    if "metric" in df.columns and "value" in df.columns:
        m = df[df["metric"].astype(str).str.lower().str.contains(n, na=False)]
        if len(m):
            return float(m.iloc[0]["value"])
    for c in df.columns:
        if n in c.lower():
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s):
                return float(s.iloc[0])
    return np.nan

def is_key_like(s):
    if s.dtype == object or pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == s.dropna().shape[0] and s.dropna().shape[0] > 1
    return False

exclude_set = set(targets) | set(excluded)
if group is not None:
    exclude_set.add(group)
feature_cols = [c for c in train.columns if c not in exclude_set and not is_key_like(train[c])]
est = final_est(latest)
names = None
for obj in [latest, est, sel, pre]:
    if hasattr(obj, "feature_names_in_"):
        names = list(obj.feature_names_in_)
        break
if names is None:
    names = feature_cols

top_feature, imp_method = "unavailable", "no_native_feature_importance"
for obj in [est, latest]:
    arr = None
    if hasattr(obj, "feature_importances_"):
        arr = np.ravel(obj.feature_importances_)
        imp_method = "feature_importances_"
    elif hasattr(obj, "coef_"):
        coef = np.asarray(obj.coef_)
        arr = np.abs(coef).mean(axis=0) if coef.ndim == 2 else np.abs(np.ravel(coef))
        imp_method = "abs_coef"
    if arr is not None and len(arr) == len(names):
        top_feature = names[int(np.argmax(arr))]
        break

sil = find_metric(metrics, primary)
params = est.get_params(deep=False) if hasattr(est, "get_params") else {}
param_keys = ["n_clusters", "init", "n_init", "max_iter", "tol", "random_state", "metric", "linkage", "affinity", "batch_size", "assign_labels", "algorithm"]
selected_params = {k: str(params[k]) for k in param_keys if k in params}

def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                out.update(flatten(v, key))
            else:
                out[key] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out

flat = flatten(alt_manifest)
manifest_params = {}
for k, v in flat.items():
    kl = k.lower()
    if any(t in kl for t in ["param", "hyper", "n_clusters", "random_state", "max_iter"]) and isinstance(v, (str, int, float, bool)):
        manifest_params[k] = str(v)
manifest_params = dict(list(sorted(manifest_params.items()))[:8])

rows = []
def add(section, item, text=np.nan, num=np.nan, source=""):
    rows.append({"section": section, "item": item, "text_value": text, "numeric_value": num, "source": source})

add("model", "frozen_model_class", est.__class__.__name__, np.nan, "latest_model")
add("model", "baseline_model_class", final_est(base).__class__.__name__, np.nan, "07_model_train_baseline_PYZ8RSVK/model.pkl")
add("model", "n_input_features", np.nan, float(len(names)), "latest_model")
add("model", "feature_schema", ", ".join(map(str, names[:20])) + ("..." if len(names) > 20 else ""), np.nan, "latest_model")
add("model", "top_feature", str(top_feature), np.nan, "latest_model")
add("model", "importance_method", imp_method, np.nan, "latest_model")
for k, v in selected_params.items():
    add("hyperparameters", k, v, np.nan, "latest_model")
for k, v in manifest_params.items():
    add("hyperparameters_manifest", k, v, np.nan, "08_model_train_alternative_3CGWKKCJ/manifest.json")
add("validation", primary, np.nan, float(sil), "09_validate_score_validation_RBGGTR3K/validation_metrics.csv")
add("inputs", "train_rows", np.nan, float(len(train)), "train")
add("inputs", "val_rows", np.nan, float(len(val)), "val")
add("limitations", "fitting", "none; frozen model interpreted only", np.nan, "interpretation.csv")
if top_feature == "unavailable":
    add("limitations", "top_feature", "model type exposes no native coefficient or feature_importance ranking", np.nan, "interpretation.csv")
add("limitations", "validation_scope", "single validation summary; no refit, threshold tuning or test use", np.nan, "interpretation.csv")

out = pd.DataFrame(rows)
mlca_runtime.save_table(out, "interpretation.csv", caption="Frozen clustering model interpretation")
if np.isfinite(sil):
    mlca_runtime.record_metric(primary, float(sil), "val")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print("frozen_model", est.__class__.__name__, "n_features", len(names))
print("top_feature", top_feature)
print("validation_silhouette", sil)
