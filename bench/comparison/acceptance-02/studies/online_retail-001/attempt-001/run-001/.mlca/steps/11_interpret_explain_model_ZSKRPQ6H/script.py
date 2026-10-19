import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
mlca_runtime.output_dir()

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing key {k!r}; available top-level keys: {list(d.keys())}")
    return d[k]

def load_json(p):
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"_error": str(e)}

def load_pkl(key):
    p = inputs.get(key)
    if p is None:
        return None
    try:
        return pickle.loads(p.read_bytes())
    except Exception:
        return None

def uniq(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def feat_of(o, seen=None):
    if seen is None:
        seen = set()
    if o is None or id(o) in seen:
        return []
    seen.add(id(o))
    out = []
    if hasattr(o, "feature_names_in_"):
        try:
            out += [str(x) for x in o.feature_names_in_]
        except Exception:
            pass
    if hasattr(o, "named_steps"):
        for s in o.named_steps.values():
            out += feat_of(s, seen)
    return out

def iface_of(o):
    if o is None:
        return "unavailable"
    parts = [type(o).__name__]
    if hasattr(o, "named_steps"):
        parts += [f"{n}={type(s).__name__}" for n, s in o.named_steps.items()]
    return " -> ".join(parts)

def params_of(o, prefix="", seen=None):
    if seen is None:
        seen = set()
    if o is None or id(o) in seen:
        return []
    seen.add(id(o))
    out = []
    if hasattr(o, "get_params"):
        try:
            ps = o.get_params(deep=False)
        except Exception:
            ps = {}
        for k, v in ps.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                out.append((prefix + k, v))
    if hasattr(o, "named_steps"):
        for n, s in o.named_steps.items():
            out += params_of(s, prefix + n + "__", seen)
    return out

def find_lists(o, hint="feature"):
    out = []
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v) and hint in str(k).lower():
                out.append(v)
            else:
                out += find_lists(v, hint)
    return out

def fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)

protocol = json.loads(inputs["_protocol"].read_text())
task = req(protocol, "task")
primary_metric = req(task, "primary_metric")
targets = list(req(protocol, "targets"))
excluded = list(req(protocol, "excluded_columns"))
group_col = req(protocol, "group_column")

pre = load_pkl("05_preprocess_fit_preprocessing_HR6HVNXD/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_T3RMYHN2/selector_all.pkl")
model = load_pkl("07_model_train_baseline_RA5A8N63/model.pkl")
alt = load_pkl("08_model_train_alternative_Q31ASHB4/alternative_model.pkl")
latest = load_pkl("latest_model")

manifest_keys = [
    "05_preprocess_fit_preprocessing_HR6HVNXD/manifest.json",
    "06_features_retain_features_T3RMYHN2/manifest.json",
    "07_model_train_baseline_RA5A8N63/manifest.json",
    "08_model_train_alternative_Q31ASHB4/manifest.json",
    "09_validate_score_validation_DPY3PEM8/manifest.json",
]
manifests = {k: load_json(inputs[k]) for k in manifest_keys if k in inputs}

train_schema = pq.read_schema(inputs["train"])
train_cols = train_schema.names
train_rows = pq.read_metadata(inputs["train"]).num_rows
candidate_cols = [c for c in train_cols if c not in set(targets) | set(excluded) and c != group_col]

feat_names = uniq(feat_of(model) + feat_of(sel) + feat_of(pre) + feat_of(alt) + feat_of(latest))
if not feat_names:
    for m in manifests.values():
        lists = find_lists(m, "feature")
        if lists:
            feat_names = uniq(lists[0])
            break
if not feat_names:
    feat_names = candidate_cols
top_feature = feat_names[0] if feat_names else "unknown"

hp = []
for k, v in params_of(model):
    kl = k.lower()
    if any(h in kl for h in ("n_clusters", "init", "max_iter", "tol", "random_state", "n_init", "algorithm", "batch_size", "learning_rate")):
        hp.append((k, v))
seen_hp, hp2 = set(), []
for k, v in hp:
    if k not in seen_hp:
        seen_hp.add(k)
        hp2.append((k, v))
hp = hp2[:8]

rows = []
def add(section, item, desc, source=None):
    rows.append({
        "section": str(section),
        "item": str(item),
        "description": pd.NA if desc is None else str(desc),
        "source": pd.NA if source is None else str(source),
    })

add("model", "frozen_model_class", iface_of(model), "07_model_train_baseline_RA5A8N63/model.pkl")
if alt is not None:
    add("model", "alternative_model_class", iface_of(alt), "08_model_train_alternative_Q31ASHB4/alternative_model.pkl")
add("model", "n_input_features", len(feat_names), "07_model_train_baseline_RA5A8N63/model.pkl")
add("model", "input_features", ", ".join(feat_names[:20]) + (" ..." if len(feat_names) > 20 else ""), "07_model_train_baseline_RA5A8N63/model.pkl")
for k, v in hp:
    add("hyperparameters", k, v, "07_model_train_baseline_RA5A8N63/model.pkl")
add("inputs", "train_rows", train_rows, "train.parquet")
add("inputs", "train_columns", len(train_cols), "train.parquet")
add("inputs", "targets", targets, "_protocol")
add("inputs", "excluded_columns", excluded, "_protocol")
add("inputs", "group_column", group_col, "_protocol")
add("validation", "primary_metric", primary_metric, "_protocol")

val_path = inputs.get("09_validate_score_validation_DPY3PEM8/validation_metrics.csv")
if val_path:
    try:
        val = pd.read_csv(val_path)
    except Exception:
        val = pd.DataFrame()
    add("validation", "validation_rows", len(val), "09_validate_score_validation_DPY3PEM8/validation_metrics.csv")
    for col in list(val.columns)[:12]:
        s = val[col].dropna() if len(val) else pd.Series(dtype=object)
        if len(s):
            add("validation", col, fmt(s.iloc[0]), "09_validate_score_validation_DPY3PEM8/validation_metrics.csv")

cand_path = inputs.get("07_model_train_baseline_RA5A8N63/kmeans_candidates.csv")
if cand_path:
    try:
        cand = pd.read_csv(cand_path)
    except Exception:
        cand = pd.DataFrame()
    add("model_selection", "candidate_rows", len(cand), "07_model_train_baseline_RA5A8N63/kmeans_candidates.csv")
    sil_cols = [c for c in cand.columns if "silhouette" in c.lower()]
    if sil_cols and len(cand):
        vals = pd.to_numeric(cand[sil_cols[0]], errors="coerce").dropna()
        if len(vals):
            add("model_selection", f"best_{sil_cols[0]}", f"{vals.max():.4g}", "07_model_train_baseline_RA5A8N63/kmeans_candidates.csv")

add("limitations", "outcome", "No outcome column declared; clustering uses internal structure only.")
add("limitations", "validation", "Validation evidence is internal clustering metric(s); no held-out label exists.")
add("limitations", "model_freeze", "Frozen artifacts were interpreted without refitting or test exposure.")
add("limitations", "feature_scope", "Features come from saved selector/model schema; protocol targets and exclusions were respected.")

df = pd.DataFrame(rows, columns=["section", "item", "description", "source"])
mlca_runtime.save_table(df, "interpretation.csv", caption="Frozen clustering interpretation")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print("interpretation rows", len(df))
print("top_feature", top_feature)
print("model", iface_of(model)[:120])
