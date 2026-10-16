import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing key {k!r}; available top-level keys: {sorted(d.keys())}")
    return d[k]

protocol = json.loads(inputs["_protocol"].read_text())
targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group = need(protocol, "group_column")
task = need(protocol, "task")
primary_metric = need(task, "primary_metric")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_keylike(s):
    if pd.api.types.is_object_dtype(s):
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == len(s)
    return False

feature_cols = [c for c in train.columns
                if c not in targets and c not in excluded and c != group
                and not is_keylike(train[c])]

def load_pkl(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

preprocessor = load_pkl("05_preprocess_fit_preprocessing_FN0Z178K/preprocessor.pkl")
selector = load_pkl("06_features_retain_features_YJWZ54M0/selector_all.pkl")
model = load_pkl("07_model_train_baseline_SMYBENGX/model.pkl")
alt_model = load_pkl("08_model_train_alternative_602XN080/alternative_model.pkl")
val_metrics = pd.read_csv(inputs["09_validate_score_validation_89GWDD0X/validation_metrics.csv"])

def final_est(obj):
    if hasattr(obj, "named_steps"):
        return list(obj.named_steps.values())[-1]
    return obj

rows = []
def add(a, d):
    rows.append({"aspect": str(a), "detail": str(d)})

add("task_type", task.get("type", "unknown"))
add("primary_metric", primary_metric)
add("n_train_rows", len(train))
add("n_val_rows", len(val))
add("n_features", len(feature_cols))
add("feature_columns", ", ".join(feature_cols[:25]) + ("..." if len(feature_cols) > 25 else ""))
add("group_column", group)
add("targets", str(targets))
add("excluded_columns", str(excluded))

for name, obj in [("baseline", model), ("alternative", alt_model)]:
    add(f"{name}_type", type(obj).__name__)
    if hasattr(obj, "named_steps"):
        add(f"{name}_steps", ", ".join(obj.named_steps.keys()))
    fe = final_est(obj)
    add(f"{name}_estimator", type(fe).__name__)
    if hasattr(fe, "get_params"):
        try:
            for k, v in fe.get_params().items():
                if k in ("memory", "verbose", "copy_X", "n_jobs", "random_state"):
                    continue
                add(f"{name}_param_{k}", v)
        except Exception:
            pass

add("validation_metrics_columns", ", ".join(val_metrics.columns))
for i, r in val_metrics.head(15).iterrows():
    add(f"validation_row_{i}", "; ".join(f"{c}={r[c]}" for c in val_metrics.columns))

add("limitation_1", "No outcome column; evaluation is internal (silhouette) only.")
add("limitation_2", "Cluster labels are not validated against external ground truth.")
add("limitation_3", "Interpretation is descriptive; no refitting was performed.")

top_feature = "unknown"
try:
    fe = final_est(model)
    if hasattr(fe, "cluster_centers_"):
        centers = np.asarray(fe.cluster_centers_)
        spread = centers.std(axis=0)
        names = feature_cols
        if hasattr(selector, "get_support"):
            try:
                support = selector.get_support()
                if len(support) == len(feature_cols):
                    names = [c for c, s in zip(feature_cols, support) if s]
            except Exception:
                pass
        if len(names) == len(spread):
            top_feature = names[int(np.argmax(spread))]
except Exception:
    pass

add("top_feature", top_feature)

interp = pd.DataFrame(rows).head(50)
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation")

mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"rows={len(interp)} features={len(feature_cols)}")
print(f"top_feature={top_feature}")
print(f"primary_metric={primary_metric}")
