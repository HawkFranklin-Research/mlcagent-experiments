import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

prot = json.loads(inputs["_protocol"].read_text())

def req(key):
    if key not in prot:
        raise KeyError(f"Missing '{key}' in _protocol. Available: {list(prot.keys())}")
    return prot[key]

targets = req("targets")
excluded = req("excluded_columns")
group_col = req("group_column")
task = req("task")
primary_metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=True) == len(s):
            continue
    if pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=True) == len(s) and (s.diff().dropna() == 1).all():
            continue
    feature_cols.append(c)

def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

preproc = load_pkl(inputs["05_preprocess_fit_preprocessing_7F4HZTDJ/preprocessor.pkl"])
selector = load_pkl(inputs["06_features_retain_features_2RG0T0WD/selector_all.pkl"])
model = load_pkl(inputs["07_model_train_baseline_8KXAGBNQ/model.pkl"])
alt_model = load_pkl(inputs["08_model_train_alternative_K6DZGSFY/alternative_model.pkl"])

baseline_manifest = json.loads(inputs["07_model_train_baseline_8KXAGBNQ/manifest.json"].read_text())
alt_manifest = json.loads(inputs["08_model_train_alternative_K6DZGSFY/manifest.json"].read_text())

def find_hp(manifest):
    if isinstance(manifest, dict):
        for k in ['hyperparameters', 'params', 'model_params']:
            if k in manifest: return manifest[k]
        for v in manifest.values():
            if isinstance(v, dict):
                hp = find_hp(v)
                if hp: return hp
    return {}

baseline_hp = find_hp(baseline_manifest)
alt_hp = find_hp(alt_manifest)

def get_val_metric(path, metric_name):
    df = pd.read_csv(path)
    if 'metric' in df.columns and 'value' in df.columns:
        sub = df[df['metric'] == metric_name]
        if 'split' in df.columns:
            sub2 = sub[sub['split'] == 'val']
            if len(sub2): sub = sub2
        if len(sub): return float(sub['value'].iloc[0])
    return None

val_metric = get_val_metric(inputs["09_validate_score_validation_DX8R1YJ4/validation_metrics.csv"], primary_metric)

def extract_top_feature(model_obj, preproc_obj, sel_obj):
    est = model_obj
    if isinstance(est, Pipeline):
        est = est.named_steps[list(est.named_steps)[-1]]
    imp = None
    if hasattr(est, 'coef_'):
        arr = np.abs(est.coef_)
        imp = arr.mean(axis=0) if arr.ndim > 1 else arr
    elif hasattr(est, 'feature_importances_'):
        imp = est.feature_importances_
    if imp is None:
        return None, None
    names = None
    for obj in [sel_obj, model_obj, preproc_obj]:
        if obj is None: continue
        try:
            names = obj.get_feature_names_out()
            if len(names) == len(imp):
                break
        except Exception:
            names = None
    if names is None:
        for obj in [model_obj, est]:
            if hasattr(obj, 'feature_names_in_'):
                names = obj.feature_names_in_
                if len(names) == len(imp):
                    break
    if names is None or len(names) != len(imp):
        return None, None
    idx = int(np.argmax(imp))
    return str(names[idx]), float(imp[idx])

top_feat, top_val = extract_top_feature(model, preproc, selector)
if top_feat is None:
    top_feat, top_val = extract_top_feature(alt_model, preproc, selector)

rows = []
def add(aspect, detail, value=np.nan, source=""):
    rows.append({"aspect": aspect, "detail": str(detail), "value": value, "source": source})

add("task_type", task.get("type", ""), np.nan, "_protocol")
add("primary_metric", primary_metric, np.nan, "_protocol")
add("targets", ",".join(targets), len(targets), "_protocol")
add("excluded_columns", ",".join(excluded), len(excluded), "_protocol")
add("group_column", group_col, np.nan, "_protocol")
add("n_train_rows", train.shape[0], train.shape[0], "train")
add("n_features", len(feature_cols), len(feature_cols), "train")
add("feature_schema", ",".join(feature_cols[:20]) + ("..." if len(feature_cols) > 20 else ""), len(feature_cols), "train")

final_est = model.named_steps[list(model.named_steps)[-1]] if isinstance(model, Pipeline) else model
add("model_class", f"{type(model).__name__} -> {type(final_est).__name__}", np.nan, "07_model_train_baseline_8KXAGBNQ/model.pkl")
add("baseline_hyperparameters", json.dumps(baseline_hp), np.nan, "07_model_train_baseline_8KXAGBNQ/manifest.json")
add("preprocessor_class", type(preproc).__name__, np.nan, "05_preprocess_fit_preprocessing_7F4HZTDJ/preprocessor.pkl")
add("selector_class", type(selector).__name__, np.nan, "06_features_retain_features_2RG0T0WD/selector_all.pkl")
add("validation_primary_metric", primary_metric, val_metric, "09_validate_score_validation_DX8R1YJ4/validation_metrics.csv")
add("top_feature", top_feat, top_val, "07_model_train_baseline_8KXAGBNQ/model.pkl")
add("limitation_repeated_subjects", f"Frames per {group_col} not independent", np.nan, "_protocol")
add("limitation_no_fit", "Frozen model; no fitting performed", np.nan, "_protocol")
add("limitation_validation_only", "Validation evidence only; no test set", np.nan, "09_validate_score_validation_DX8R1YJ4/validation_metrics.csv")

if alt_model is not None:
    alt_final = alt_model.named_steps[list(alt_model.named_steps)[-1]] if isinstance(alt_model, Pipeline) else alt_model
    add("alternative_model_class", f"{type(alt_model).__name__} -> {type(alt_final).__name__}", np.nan, "08_model_train_alternative_K6DZGSFY/alternative_model.pkl")
    add("alternative_hyperparameters", json.dumps(alt_hp), np.nan, "08_model_train_alternative_K6DZGSFY/manifest.json")

for name, obj, src in [
    ("preprocessor", preproc, "05_preprocess_fit_preprocessing_7F4HZTDJ/preprocessor.pkl"),
    ("selector", selector, "06_features_retain_features_2RG0T0WD/selector_all.pkl"),
    ("model", model, "07_model_train_baseline_8KXAGBNQ/model.pkl")
]:
    if hasattr(obj, "feature_names_in_"):
        fn = list(obj.feature_names_in_)
        match = fn == feature_cols
        add(f"feature_names_in_{name}", f"n={len(fn)}, match_raw_schema={match}", np.nan, src)

rows = rows[:50]
interp_df = pd.DataFrame(rows)
interp_df["value"] = interp_df["value"].apply(
    lambda x: float(f"{x:.4g}") if pd.notna(x) and isinstance(x, (int, float)) else x
)

mlca_runtime.save_table(interp_df, "interpretation.csv", caption="Frozen model interpretation: hyperparameters, inputs, validation and limitations.")

if top_feat is not None:
    mlca_runtime.record_fact_candidate("top_feature", top_feat, source="07_model_train_baseline_8KXAGBNQ/model.pkl")

print("Interpretation rows:", len(interp_df))
print("Top feature:", top_feat)
