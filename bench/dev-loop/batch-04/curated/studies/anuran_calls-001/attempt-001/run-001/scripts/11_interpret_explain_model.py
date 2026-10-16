import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()

def load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def unwrap(obj):
    if isinstance(obj, dict):
        for k in ["model", "estimator", "pipeline", "best_model"]:
            if k in obj:
                return obj[k]
    return obj

def get_protocol_key(protocol, key):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}. Available top-level keys: {list(protocol.keys())}")
    return protocol[key]

protocol = json.loads(paths["_protocol"].read_text())
targets = get_protocol_key(protocol, "targets")
excluded = get_protocol_key(protocol, "excluded_columns")
group_col = get_protocol_key(protocol, "group_column")
task = get_protocol_key(protocol, "task")
if "primary_metric" not in task:
    raise KeyError(f"Missing task key 'primary_metric'. Available task keys: {list(task.keys())}")
primary_metric = task["primary_metric"]

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

def is_key_like(s):
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == len(s)
    return False

feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group_col and not is_key_like(train[c])
]

preprocessor = load_pickle(paths["05_preprocess_fit_preprocessing_XYF9R0M6/preprocessor.pkl"])
selector = load_pickle(paths["06_features_retain_features_P9FZ0GCB/selector_all.pkl"])
baseline = unwrap(load_pickle(paths["07_model_train_baseline_SGXK026F/model.pkl"]))
latest = unwrap(load_pickle(paths["latest_model"]))
alt = unwrap(load_pickle(paths["08_model_train_alternative_MK5K74KS/alternative_model.pkl"]))

def final_est(obj):
    if isinstance(obj, Pipeline):
        steps = obj.named_steps
        if steps:
            return steps[list(steps)[-1]]
    return obj

def feature_names(obj, fallback):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        first = obj.named_steps[list(obj.named_steps)[0]]
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
        if hasattr(first, "get_feature_names_out"):
            try:
                return list(first.get_feature_names_out())
            except Exception:
                pass
    if hasattr(obj, "get_feature_names_out"):
        try:
            return list(obj.get_feature_names_out())
        except Exception:
            pass
    return fallback

def top_feature(model, fallback):
    final = final_est(model)
    ests = list(final.estimators_) if hasattr(final, "estimators_") else [final]
    imps = []
    for e in ests:
        if hasattr(e, "coef_"):
            c = np.asarray(e.coef_)
            imps.append(np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c))
        elif hasattr(e, "feature_importances_"):
            imps.append(np.asarray(e.feature_importances_))
    if not imps:
        return None
    n = min(len(i) for i in imps)
    imp = np.mean([i[:n] for i in imps], axis=0)
    idx = int(np.argmax(imp))
    names = feature_names(model, fallback)
    return names[idx] if idx < len(names) else f"feature_{idx}"

def params_summary(model):
    final = final_est(model)
    if hasattr(final, "estimator"):
        final = final.estimator
    try:
        p = final.get_params()
    except Exception:
        return {}
    keys = [
        "n_estimators", "max_depth", "learning_rate", "C", "penalty", "solver",
        "alpha", "max_iter", "min_samples_leaf", "min_samples_split",
        "class_weight", "random_state", "loss", "hidden_layer_sizes", "activation"
    ]
    out = {k: p[k] for k in keys if k in p}
    return out or dict(list(p.items())[:10])

selected_features = feature_names(selector, feature_cols)
top = top_feature(latest, selected_features) or top_feature(baseline, selected_features) or "unknown"

val_metrics = pd.read_csv(paths["09_validate_score_validation_DGS5S5Y8/validation_metrics.csv"])
if "metric" in val_metrics.columns and "value" in val_metrics.columns:
    row = val_metrics[val_metrics["metric"] == primary_metric]
    val_score = row["value"].iloc[0] if len(row) else val_metrics["value"].iloc[0]
elif primary_metric in val_metrics.columns:
    val_score = val_metrics[primary_metric].iloc[0]
else:
    num = val_metrics.select_dtypes(include=[np.number]).columns
    val_score = val_metrics[num[0]].iloc[0] if len(num) else np.nan
val_score = pd.to_numeric(val_score, errors="coerce")

def num4(x):
    return float(f"{x:.4g}") if pd.notna(x) else np.nan

def row_text(item, value, source):
    return {"item": item, "value_text": value, "value_numeric": np.nan, "source": source}

def row_num(item, value, source):
    return {"item": item, "value_text": np.nan, "value_numeric": value, "source": source}

rows = [
    row_text("model_type", type(latest).__name__, "latest_model"),
    row_text("baseline_model_type", type(baseline).__name__, "07_model_train_baseline_SGXK026F/model.pkl"),
    row_text("alternative_model_type", type(alt).__name__, "08_model_train_alternative_MK5K74KS/alternative_model.pkl"),
    row_text("preprocessor_type", type(preprocessor).__name__, "05_preprocess_fit_preprocessing_XYF9R0M6/preprocessor.pkl"),
    row_text("selector_type", type(selector).__name__, "06_features_retain_features_P9FZ0GCB/selector_all.pkl"),
    row_text("targets", ", ".join(targets), "_protocol"),
    row_text("primary_metric", primary_metric, "_protocol"),
    row_num("feature_count", len(feature_cols), "train.parquet"),
    row_num("train_rows", len(train), "train.parquet"),
    row_num("val_rows", len(val), "val.parquet"),
    row_num("validation_score", num4(val_score), "09_validate_score_validation_DGS5S5Y8/validation_metrics.csv"),
    row_text("top_feature", top, "latest_model"),
    row_text("hyperparameters", json.dumps(params_summary(latest), default=str), "latest_model"),
    row_text(
        "limitations",
        "Frozen model not refit; validation-only evidence; RecordID repeats across rows; "
        "multiple MFCC vectors per recording; hierarchical targets may be correlated; no test evaluation.",
        "interpretation.csv"
    ),
]

interp = pd.DataFrame(rows)
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation")
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")

print(f"train rows: {len(train)}, val rows: {len(val)}")
print(f"features: {len(feature_cols)}, targets: {len(targets)}")
print(f"primary metric: {primary_metric}, validation score: {val_score}")
print(f"top feature: {top}")
print("saved interpretation.csv")
