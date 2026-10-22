import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key}; available top-level keys: {sorted(protocol.keys())}")
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task['primary_metric']; available top-level keys: {sorted(protocol.keys())}; task keys: {sorted(task.keys())}")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]
primary_metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
def is_key_like(s, name):
    if s.nunique(dropna=True) != len(s):
        return False
    if s.dtype.kind in "OUS" or str(s.dtype) in ("object", "string", "category"):
        return True
    return any(tok in str(name).lower() for tok in ["id", "code", "index", "row", "key", "serial"])

feature_cols = [c for c in train.columns if c not in set(targets + excluded + [group]) and not is_key_like(train[c], c)]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model_src = "07_model_train_baseline_SWTMFPFT/model.pkl"
alt_src = "08_model_train_alternative_XQ00QFTN/alternative_model.pkl"
val_src = "09_validate_score_validation_H8CSBPBR/validation_metrics.csv"
cand_src = "07_model_train_baseline_SWTMFPFT/cluster_candidates.csv"
model = load_pkl(inputs[model_src])
alt = load_pkl(inputs[alt_src])
pre = load_pkl(inputs["05_preprocess_fit_preprocessing_1E8V578C/preprocessor.pkl"])
sel = load_pkl(inputs["06_features_retain_features_Y7PF8JJ2/selector_all.pkl"])

def final_est(obj):
    if hasattr(obj, "named_steps") and obj.named_steps:
        return list(obj.named_steps.values())[-1]
    return obj

final = final_est(model)
def output_names(n):
    for o in [sel, pre, model]:
        if hasattr(o, "get_feature_names_out"):
            try:
                names = list(o.get_feature_names_out())
                if len(names) == n:
                    return names
            except Exception:
                pass
    for o in [model, sel, pre]:
        if hasattr(o, "feature_names_in_"):
            names = list(o.feature_names_in_)
            if len(names) == n:
                return names
    return None

nclusters = getattr(final, "n_clusters", np.nan)
if hasattr(final, "cluster_centers_"):
    centers = np.asarray(final.cluster_centers_, dtype=float)
    nfeat = centers.shape[1]
    names = output_names(nfeat) or [f"feature_{i}" for i in range(nfeat)]
    top_feature = names[int(np.argmax(centers.std(axis=0)))]
    nclusters = centers.shape[0]
elif hasattr(final, "feature_importances_"):
    vals = np.asarray(final.feature_importances_, dtype=float)
    nfeat = len(vals)
    names = output_names(nfeat) or [f"feature_{i}" for i in range(nfeat)]
    top_feature = names[int(np.argmax(np.abs(vals)))]
elif hasattr(final, "coef_"):
    vals = np.asarray(final.coef_, dtype=float)
    vals = np.abs(vals).mean(axis=0) if vals.ndim > 1 else np.abs(vals)
    nfeat = len(vals)
    names = output_names(nfeat) or [f"feature_{i}" for i in range(nfeat)]
    top_feature = names[int(np.argmax(vals))]
else:
    nfeat = len(feature_cols)
    top_feature = feature_cols[0] if feature_cols else "unknown"
    try:
        support = np.asarray(sel.get_support())
        if len(feature_cols) == len(support) and support.any():
            top_feature = feature_cols[int(np.where(support)[0][0])]
    except Exception:
        pass

def metric_value(df, name):
    for c in df.columns:
        if name in c.lower():
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s):
                return float(s.iloc[0])
    lc = {c.lower(): c for c in df.columns}
    for key in ["metric", "name", "metric_name"]:
        if key in lc:
            for vc in ["value", "score", "val"]:
                if vc in lc:
                    mask = df[lc[key]].astype(str).str.lower().str.contains(name)
                    if mask.any():
                        s = pd.to_numeric(df.loc[mask, lc[vc]], errors="coerce").dropna()
                        if len(s):
                            return float(s.iloc[0])
    return np.nan

def best_candidate(df):
    scol = next((c for c in df.columns if "silhouette" in c.lower()), None)
    if scol is None:
        return np.nan, np.nan
    s = pd.to_numeric(df[scol], errors="coerce")
    if not s.notna().any():
        return np.nan, np.nan
    idx = s.idxmax()
    kcol = next((c for c in df.columns if any(t in c.lower() for t in ["cluster", "n_clusters", "k"])), None)
    return (df.loc[idx, kcol] if kcol else np.nan), float(s.loc[idx])

val = pd.read_csv(inputs[val_src])
cand = pd.read_csv(inputs[cand_src])
sil = metric_value(val, "silhouette")
best_k, best_s = best_candidate(cand)

def num4(x):
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(f"{x:.4g}") if np.isfinite(x) else np.nan
    return x

rows = []
def add(section, item, value, source):
    if value is None:
        value = np.nan
    if isinstance(value, str) and value == "":
        value = np.nan
    if isinstance(value, (list, tuple, dict)):
        value = str(value)
    rows.append({"section": section, "item": item, "value": num4(value), "source": source})

add("model", "selected_model_file", model_src, model_src)
add("model", "estimator", type(final).__name__, model_src)
add("model", "pipeline_steps", " > ".join(model.named_steps.keys()) if hasattr(model, "named_steps") else np.nan, model_src)
add("model", "alternative_estimator", type(final_est(alt)).__name__, alt_src)
add("model", "n_clusters", nclusters, model_src)
add("model", "top_feature", top_feature, "interpretation.csv")
add("inputs", "feature_columns", ", ".join(feature_cols), "train.parquet")
add("inputs", "n_input_features", len(feature_cols), "train.parquet")
add("inputs", "n_model_features", nfeat, model_src)

params = final.get_params(deep=False) if hasattr(final, "get_params") else {}
preferred = ["n_clusters", "init", "n_init", "max_iter", "tol", "algorithm", "random_state", "metric", "linkage", "affinity", "batch_size", "assign_labels"]
hp = []
for k in preferred:
    if k in params:
        hp.append((k, params[k]))
for k, v in params.items():
    if len(hp) >= 12:
        break
    if k not in [x[0] for x in hp] and isinstance(v, (str, int, float, bool, type(None), np.integer, np.floating)):
        hp.append((k, v))
for k, v in hp:
    add("hyperparameters", k, v, model_src)

add("validation", "primary_metric", primary_metric, "_protocol")
add("validation", "validation_silhouette", sil, val_src)
add("validation", "best_candidate_k", best_k, cand_src)
add("validation", "best_candidate_silhouette", best_s, cand_src)
add("validation", "candidate_rows", len(cand), cand_src)

for item, text in [
    ("no_outcome", "No outcome or label column is present; clusters are unsupervised segmentation summaries."),
    ("internal_metric", "Validation evidence is internal silhouette only and may not reflect external validity."),
    ("no_refit", "The frozen model and preprocessing were described without fitting or refitting on validation rows."),
    ("scaling", "Distance-based clustering uses scaled features, so top_feature reflects standardized model space."),
    ("top_feature", "top_feature is derived from the frozen estimator's dispersion or importance and is not causal."),
    ("split", "Validation evidence is limited to the declared validation partition used as given."),
]:
    add("limitations", item, text, "interpretation.csv")

out = pd.DataFrame(rows, columns=["section", "item", "value", "source"])
mlca_runtime.save_table(out, "interpretation.csv", caption="Frozen clustering model interpretation")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print("features", len(feature_cols))
print("model", type(final).__name__, "clusters", nclusters)
print("validation_silhouette", num4(sil))
print("top_feature", top_feature)
print("wrote interpretation.csv")
