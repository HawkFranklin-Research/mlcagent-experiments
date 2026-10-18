import mlca_runtime
import json, pickle, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
needed = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in needed if k not in protocol]
if missing:
    raise KeyError(f"Missing _protocol keys {missing}; available top-level keys: {list(protocol.keys())}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; task keys: {list(task.keys())}")
primary_metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])

def load_pkl(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

preprocessor = load_pkl("05_preprocess_fit_preprocessing_XD81KYMB/preprocessor.pkl")
selector = load_pkl("06_features_retain_features_75C8096C/selector_all.pkl")
model = load_pkl("07_model_train_baseline_93EYVTD0/model.pkl")
alt = load_pkl("08_model_train_alternative_NZ42FXM6/alternative_model.pkl")

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or (group_col is not None and c == group_col):
        continue
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
        if any(t in str(c).lower() for t in ["id", "code", "key", "index"]):
            continue
    feature_cols.append(c)
if not feature_cols:
    raise ValueError("No feature columns left after protocol exclusions")

def step_summary(obj):
    if isinstance(obj, Pipeline):
        return type(obj).__name__ + "[" + ", ".join(f"{k}:{type(v).__name__}" for k, v in obj.named_steps.items()) + "]"
    return type(obj).__name__

def final_est(obj):
    while isinstance(obj, Pipeline) and obj.named_steps:
        obj = list(obj.named_steps.values())[-1]
    return obj

KEYS = {"n_components", "whiten", "svd_solver", "tol", "iterated_power", "random_state",
        "hidden_layer_sizes", "activation", "solver", "alpha", "learning_rate_init",
        "max_iter", "batch_size", "beta_1", "beta_2", "epsilon", "n_iter_no_change",
        "early_stopping", "validation_fraction"}

def simple_params(est):
    out = {}
    try:
        params = est.get_params(deep=False)
    except Exception:
        params = {}
    for k, v in params.items():
        if k in KEYS:
            out[k] = v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
    if not out:
        for k in ["n_components", "whiten", "svd_solver"]:
            if hasattr(est, k):
                out[k] = getattr(est, k)
    return out

def find_pca(obj):
    if hasattr(obj, "components_") and hasattr(obj, "transform"):
        return obj
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            r = find_pca(step)
            if r is not None:
                return r
    return None

def selector_names(sel):
    if hasattr(sel, "get_support"):
        try:
            sup = sel.get_support()
            if len(sup) == len(feature_cols):
                return [c for c, s in zip(feature_cols, sup) if s]
        except Exception:
            pass
    if hasattr(sel, "get_feature_names_out"):
        try:
            return list(sel.get_feature_names_out(feature_cols))
        except Exception:
            pass
    if hasattr(sel, "feature_names_in_"):
        return list(sel.feature_names_in_)
    return None

top_feature = None
pca = find_pca(model) or find_pca(alt)
if pca is not None and hasattr(pca, "components_"):
    comp = np.asarray(pca.components_)
    if comp.ndim == 2 and comp.shape[0] > 0 and comp.shape[1] > 0:
        idx = int(np.argmax(np.abs(comp[0])))
        names = None
        if hasattr(pca, "feature_names_in_") and len(pca.feature_names_in_) == comp.shape[1]:
            names = list(pca.feature_names_in_)
        if names is None:
            sn = selector_names(selector)
            if sn is not None and len(sn) == comp.shape[1]:
                names = sn
        if names is None and len(feature_cols) == comp.shape[1]:
            names = feature_cols
        if names is not None and idx < len(names):
            top_feature = str(names[idx])
if top_feature is None:
    variances = train[feature_cols].var(numeric_only=True)
    top_feature = str(variances.idxmax()) if len(variances) and variances.notna().any() else str(feature_cols[0])

val_path = inputs["09_validate_score_validation_TA0EMQEK/validation_metrics.csv"]
val = pd.read_csv(val_path)
items = []
if {"metric", "value"}.issubset(val.columns):
    for _, row in val.iterrows():
        if pd.notna(row["value"]):
            split = row["split"] if "split" in val.columns and pd.notna(row["split"]) else ""
            items.append(f"{split}:{row['metric']}={float(row['value']):.4g}")
elif "reconstruction_error" in val.columns:
    for v in val["reconstruction_error"].dropna().head(5):
        items.append(f"reconstruction_error={float(v):.4g}")
else:
    for c in val.select_dtypes(include="number").columns[:5]:
        if len(val[c].dropna()):
            items.append(f"{c}={float(val[c].dropna().iloc[0]):.4g}")
metrics_str = "; ".join(items) if items else "not available"

rows = [
    ("task_type", str(task.get("type", "")), "_protocol"),
    ("primary_metric", str(primary_metric), "_protocol"),
    ("targets", str(targets), "_protocol"),
    ("excluded_columns", str(excluded), "_protocol"),
    ("group_column", str(group_col), "_protocol"),
    ("feature_count_used", str(len(feature_cols)), "train.parquet + _protocol"),
    ("feature_schema", ", ".join(feature_cols[:6]) + (" ..." if len(feature_cols) > 6 else ""), "train.parquet"),
    ("preprocessor", step_summary(preprocessor), "05_preprocess_fit_preprocessing_XD81KYMB/preprocessor.pkl"),
    ("selector", step_summary(selector), "06_features_retain_features_75C8096C/selector_all.pkl"),
    ("baseline_model", step_summary(model), "07_model_train_baseline_93EYVTD0/model.pkl"),
    ("alternative_model", step_summary(alt), "08_model_train_alternative_NZ42FXM6/alternative_model.pkl"),
    ("baseline_final_estimator", type(final_est(model)).__name__, "07_model_train_baseline_93EYVTD0/model.pkl"),
    ("alternative_final_estimator", type(final_est(alt)).__name__, "08_model_train_alternative_NZ42FXM6/alternative_model.pkl"),
    ("baseline_hyperparameters", json.dumps(simple_params(final_est(model)), default=str), "07_model_train_baseline_93EYVTD0/model.pkl"),
    ("alternative_hyperparameters", json.dumps(simple_params(final_est(alt)), default=str), "08_model_train_alternative_NZ42FXM6/alternative_model.pkl"),
    ("validation_evidence", metrics_str, "09_validate_score_validation_TA0EMQEK/validation_metrics.csv"),
    ("limitations", "Unsupervised representation with no outcome/target; validation evidence is reconstruction error only; no supervised feature ranking; frozen models were not refit; conclusions are limited to input pixel intensities and this validation split.", "_protocol + 09_validate_score_validation_TA0EMQEK/validation_metrics.csv"),
    ("top_feature", top_feature, "interpretation.csv"),
]
interp = pd.DataFrame(rows, columns=["item", "value", "source"])
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen representation model interpretation")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"features={len(feature_cols)}; baseline={type(final_est(model)).__name__}; top_feature={top_feature}")
print(f"validation={metrics_str[:180]}")
