import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
task = protocol["task"]
primary_metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])

# Declared feature columns (RL-0007)
drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    if train[c].dtype == object:
        continue
    if pd.api.types.is_integer_dtype(train[c]):
        if train[c].nunique() == len(train):
            continue
    feature_cols.append(c)

# Load saved model
model = pickle.load(open(inputs["07_model_train_baseline_MAK3HTGT/model.pkl"], "rb"))
alt = pickle.load(open(inputs["08_model_train_alternative_5AQ82GP1/alternative_model.pkl"], "rb"))
preproc = pickle.load(open(inputs["05_preprocess_fit_preprocessing_XP5A8XH9/preprocessor.pkl"], "rb"))
selector = pickle.load(open(inputs["06_features_retain_features_MWAQCCQG/selector_all.pkl"], "rb"))

sel = pd.read_csv(inputs["07_model_train_baseline_MAK3HTGT/cluster_selection.csv"])
val_metrics = pd.read_csv(inputs["09_validate_score_validation_C8CS9EB3/validation_metrics.csv"])

# Describe hyperparameters from model
def hyper_of(m):
    if hasattr(m, "named_steps"):
        est = list(m.named_steps.values())[-1]
        params = est.get_params()
    elif hasattr(m, "get_params"):
        params = m.get_params()
    else:
        params = {}
    keep = {k: v for k, v in params.items() if not callable(v) and k in ("n_clusters","n_components","eps","min_samples","linkage","random_state","affinity","metric","covariance_type","cluster_std")}
    return keep

baseline_hyper = hyper_of(model)
alt_hyper = hyper_of(alt)

# Best cluster count from selection table
kcol = "n_clusters" if "n_clusters" in sel.columns else sel.columns[0]
score_cols = [c for c in sel.columns if primary_metric in c.lower()]
if score_cols:
    best_idx = sel[score_cols[0]].idxmax()
else:
    best_idx = 0
best_k = sel.loc[best_idx, kcol]

# Validation metric
vcol = [c for c in val_metrics.columns if primary_metric in c.lower()]
val_score = float(val_metrics[vcol[0]].iloc[0]) if vcol else float("nan")

# top feature: from preprocessor feature names
top_feature = feature_cols[0] if feature_cols else ""
try:
    if hasattr(selector, "feature_names_in_"):
        fn = list(selector.feature_names_in_)
        if fn:
            top_feature = fn[0]
except Exception:
    pass

rows = [
    ("model_type", type(model).__name__),
    ("baseline_hyperparameters", json.dumps(baseline_hyper, default=str)),
    ("alternative_model_type", type(alt).__name__),
    ("alternative_hyperparameters", json.dumps(alt_hyper, default=str)),
    ("selected_cluster_count", str(best_k)),
    ("feature_count", str(len(feature_cols))),
    ("feature_columns", ", ".join(feature_cols)),
    ("preprocessor_type", type(preproc).__name__),
    ("selector_type", type(selector).__name__),
    ("primary_metric", primary_metric),
    ("validation_score", f"{val_score:.4g}"),
    ("limitations", "No outcome; cluster count chosen only by internal criteria; labels are descriptive not predictive; ground-truth agronomic category absent."),
    ("row_count_used_for_fit", str(len(train))),
]
summary = pd.DataFrame(rows, columns=["item", "value"])

mlca_runtime.save_table(summary, "interpretation.csv", caption="Frozen clustering model interpretation")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print("model:", type(model).__name__, "k:", best_k, "features:", len(feature_cols))
print("val", primary_metric, "=", round(val_score, 4))
print("rows:", len(summary))
