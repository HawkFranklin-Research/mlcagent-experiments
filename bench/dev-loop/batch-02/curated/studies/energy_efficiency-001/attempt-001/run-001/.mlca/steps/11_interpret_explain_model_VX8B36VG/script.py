import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys: {missing}. Available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]
task = protocol["task"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

all_cols = list(train.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]
key_like = []
for c in feature_cols:
    s = train[c]
    if s.dtype == object:
        key_like.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique() == len(s):
        key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

preprocessor = load_pkl(inputs["05_preprocess_fit_preprocessing_K5VFSMR2/preprocessor.pkl"])
selector = load_pkl(inputs["06_features_retain_features_8F2KVPF9/selector_all.pkl"])
baseline_model = load_pkl(inputs["07_model_train_baseline_0WM3R8E0/model.pkl"])
alt_model = load_pkl(inputs["08_model_train_alternative_B8FQJ4R0/alternative_model.pkl"])
latest_model = load_pkl(inputs["latest_model"])

def load_json(path):
    with open(path) as f:
        return json.load(f)

pre_manifest = load_json(inputs["05_preprocess_fit_preprocessing_K5VFSMR2/manifest.json"])
sel_manifest = load_json(inputs["06_features_retain_features_8F2KVPF9/manifest.json"])
base_manifest = load_json(inputs["07_model_train_baseline_0WM3R8E0/manifest.json"])
alt_manifest = load_json(inputs["08_model_train_alternative_B8FQJ4R0/manifest.json"])
val_manifest = load_json(inputs["09_validate_score_validation_8SYXFK0A/manifest.json"])

val_metrics = pd.read_csv(inputs["09_validate_score_validation_8SYXFK0A/validation_metrics.csv"])

rows = []
def add(aspect, detail, source):
    rows.append({"aspect": aspect, "detail": str(detail), "source": source})

add("task_type", task.get("type", "unknown"), "_protocol")
add("primary_metric", task.get("primary_metric", "unknown"), "_protocol")
add("targets", ", ".join(targets), "_protocol")
add("excluded_columns", ", ".join(excluded) if excluded else "none", "_protocol")
add("group_column", group_col if group_col else "none", "_protocol")
add("n_features", len(feature_cols), "train.parquet")
add("feature_columns", ", ".join(feature_cols), "train.parquet")
add("key_like_columns_removed", ", ".join(key_like) if key_like else "none", "train.parquet")

def describe_model(name, model, manifest, source):
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.keys())
        add(f"{name}_pipeline_steps", ", ".join(steps), source)
        final_step = model.named_steps[steps[-1]]
        if hasattr(final_step, "get_params"):
            params = final_step.get_params()
            for k, v in params.items():
                if v is not None and k not in ("verbose", "random_state"):
                    add(f"{name}_hyperparam_{k}", str(v), source)
    else:
        add(f"{name}_type", type(model).__name__, source)
        if hasattr(model, "get_params"):
            params = model.get_params()
            for k, v in params.items():
                if v is not None and k not in ("verbose", "random_state"):
                    add(f"{name}_hyperparam_{k}", str(v), source)
    if manifest:
        for k, v in manifest.items():
            if k not in ("feature_names", "feature_names_in_", "selected_features"):
                add(f"{name}_manifest_{k}", str(v), source)

describe_model("baseline", baseline_model, base_manifest, "07_model_train_baseline_0WM3R8E0/manifest.json")
describe_model("alternative", alt_model, alt_manifest, "08_model_train_alternative_B8FQJ4R0/manifest.json")
describe_model("latest", latest_model, alt_manifest, "latest_model")

for obj_name, obj, src in [("preprocessor", preprocessor, "05_preprocess_fit_preprocessing_K5VFSMR2/preprocessor.pkl"),
                           ("selector", selector, "06_features_retain_features_8F2KVPF9/selector_all.pkl"),
                           ("baseline", baseline_model, "07_model_train_baseline_0WM3R8E0/model.pkl"),
                           ("alternative", alt_model, "08_model_train_alternative_B8FQJ4R0/alternative_model.pkl"),
                           ("latest", latest_model, "latest_model")]:
    if hasattr(obj, "feature_names_in_"):
        add(f"{obj_name}_feature_names_in_", ", ".join(obj.feature_names_in_), src)

add("validation_metrics", json.dumps(val_metrics.to_dict(orient='records')), "09_validate_score_validation_8SYXFK0A/validation_metrics.csv")

add("limitation", "No test set evaluation; validation only.", "09_validate_score_validation_8SYXFK0A/manifest.json")
add("limitation", "Model trained on simulated building configurations; may not generalize to real buildings.", "train.parquet")
add("limitation", "No time or group structure; independent rows assumed.", "_protocol")

top_feature = None
top_source = ""
if "selected_features" in sel_manifest:
    feats = sel_manifest["selected_features"]
    if feats:
        top_feature = feats[0]
        top_source = "06_features_retain_features_8F2KVPF9/manifest.json"
else:
    sel = selector
    if hasattr(sel, "named_steps"):
        for step in reversed(list(sel.named_steps.values())):
            if hasattr(step, "get_support"):
                sel = step
                break
    if hasattr(sel, "get_support") and hasattr(sel, "feature_names_in_"):
        support = sel.get_support()
        feats = sel.feature_names_in_[support]
        if len(feats) > 0:
            top_feature = feats[0]
            top_source = "06_features_retain_features_8F2KVPF9/selector_all.pkl"
if top_feature is None:
    top_feature = feature_cols[0] if feature_cols else "unknown"
    top_source = "train.parquet"

add("top_feature", top_feature, top_source)

interp_df = pd.DataFrame(rows)
interp_df = interp_df.head(50)
mlca_runtime.save_table(interp_df, "interpretation.csv", caption="Frozen model interpretation")

mlca_runtime.record_fact_candidate("top_feature", top_feature, source=top_source)

print(f"Features: {len(feature_cols)}")
print(f"Targets: {targets}")
print(f"Validation metrics rows: {len(val_metrics)}")
print(f"Top feature: {top_feature}")
print("Interpretation saved.")
