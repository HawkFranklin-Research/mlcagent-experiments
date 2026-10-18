import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    group = protocol["group_column"]
    task = protocol["task"]
    metric = task["primary_metric"]
except KeyError as exc:
    raise KeyError(f"Missing protocol key {exc}; available keys: {sorted(protocol)}") from exc

if isinstance(targets, str):
    targets = [targets]
if isinstance(excluded, str):
    excluded = [excluded]

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

excluded_set = set(targets) | set(excluded) | ({group} if group is not None else set())
feature_cols = []
for col in train.columns:
    if col in excluded_set:
        continue
    s = train[col]
    is_text = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)
    is_int = pd.api.types.is_integer_dtype(s)
    if (is_text or is_int) and s.nunique(dropna=False) == len(s):
        continue
    feature_cols.append(col)

X_val = val[feature_cols].copy()
y_val = val[targets[0]]

def load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)

pre = load_pickle(paths["06_preprocess_fit_preprocessing_MYPQXFW6/preprocessor.pkl"])
sel = load_pickle(paths["07_features_select_features_NK102NGP/selector.pkl"])
model = load_pickle(paths["08_model_train_logistic_GP5KS1CK/model.pkl"])
latest = load_pickle(paths["latest_model"])

def accepts_raw_features(obj):
    if not hasattr(obj, "predict_proba"):
        return False
    try:
        obj.predict_proba(X_val.head(2))
        return True
    except Exception:
        return False

estimator = None
for candidate in [latest, model]:
    if accepts_raw_features(candidate):
        estimator = candidate
        break

if estimator is None:
    frontend = None
    if sel is not None and hasattr(sel, "transform"):
        try:
            sel.transform(X_val.head(2))
            frontend = sel
        except Exception:
            frontend = None
    if frontend is None:
        steps = []
        if pre is not None:
            steps += list(pre.steps) if isinstance(pre, Pipeline) else [("preprocessor", pre)]
        if sel is not None:
            steps += list(sel.steps) if isinstance(sel, Pipeline) else [("selector", sel)]
        frontend = Pipeline(steps) if steps else None
    estimator = Pipeline([("frontend", frontend), ("model", model)]) if frontend is not None else model
    if not accepts_raw_features(estimator):
        raise RuntimeError("No complete saved pipeline accepts the raw feature DataFrame")

scoring = metric if metric in {"roc_auc", "roc_auc_ovr", "roc_auc_ovo", "roc_auc_ovr_weighted", "roc_auc_ovo_weighted"} else "roc_auc_ovr"
result = permutation_importance(
    estimator,
    X_val,
    y_val,
    scoring=scoring,
    n_repeats=20,
    random_state=42,
    n_jobs=1,
)

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": result.importances_mean,
    "importance_sd": result.importances_std,
}).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
importance["importance_mean"] = importance["importance_mean"].map(lambda v: float(f"{v:.4g}"))
importance["importance_sd"] = importance["importance_sd"].map(lambda v: float(f"{v:.4g}"))

mlca_runtime.save_table(importance, "interpretation.csv", caption="Top 15 permutation importances on validation rows")
top_feature = importance.loc[0, "feature"]
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"features={len(feature_cols)} val_rows={len(X_val)}")
print(f"top_feature={top_feature}")
