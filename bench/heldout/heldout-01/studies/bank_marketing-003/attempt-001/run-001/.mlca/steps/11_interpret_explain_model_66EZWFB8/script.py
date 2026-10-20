import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing {key}; available top-level keys: {sorted(protocol)}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task.primary_metric; task keys: {sorted(task)}")
primary = task["primary_metric"]

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

drop = set(targets) | set(excluded) | ({group} if group else set())
features = [c for c in train.columns if c not in drop]
target_col = targets[0]
y_val = val[target_col]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def is_complete(obj, feats):
    return (
        isinstance(obj, Pipeline)
        and hasattr(obj, "feature_names_in_")
        and set(obj.feature_names_in_) == set(feats)
    )

latest = load_pkl(paths["latest_model"])
if is_complete(latest, features):
    est = latest
else:
    model07 = load_pkl(paths["07_model_train_logistic_VP57BH50/model.pkl"])
    if is_complete(model07, features):
        est = model07
    else:
        pre = load_pkl(paths["05_preprocess_fit_preprocessing_KGJNVG5D/preprocessor.pkl"])
        sel = load_pkl(paths["06_features_select_features_20NTEHHR/selector.pkl"])
        mdl = model07
        if isinstance(sel, Pipeline) and hasattr(sel, "feature_names_in_") and set(sel.feature_names_in_) == set(features):
            est = Pipeline([("selector", sel), ("model", mdl)])
        elif isinstance(pre, Pipeline) and hasattr(pre, "feature_names_in_") and set(pre.feature_names_in_) == set(features):
            est = Pipeline([("preprocessor", pre), ("selector", sel), ("model", mdl)])
        else:
            raise RuntimeError("Could not build a frozen pipeline accepting the raw feature columns")

if hasattr(est, "feature_names_in_"):
    cols = [c for c in est.feature_names_in_ if c in features]
    X_val = val[cols]
else:
    cols = features
    X_val = val[cols]

if primary == "roc_auc" and task.get("n_classes", 2) > 2:
    scoring = "roc_auc_ovr"
elif primary == "roc_auc":
    scoring = "roc_auc"
else:
    scoring = primary

res = permutation_importance(
    est,
    X_val,
    y_val,
    scoring=scoring,
    n_repeats=20,
    random_state=42,
    n_jobs=-1,
)

imp = (
    pd.DataFrame(
        {
            "feature": cols,
            "importance_mean": res.importances_mean,
            "importance_sd": res.importances_std,
        }
    )
    .sort_values("importance_mean", ascending=False)
    .head(15)
    .reset_index(drop=True)
    .round(4)
)

mlca_runtime.save_table(
    imp,
    "interpretation.csv",
    caption="Permutation importance on validation rows",
)
top = str(imp.iloc[0]["feature"])
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")

print(f"train rows: {len(train)}")
print(f"val rows: {len(val)}")
print(f"top feature: {top}")
