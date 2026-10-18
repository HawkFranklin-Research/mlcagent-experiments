import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
p = json.loads(inputs["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing_keys = [k for k in required if k not in p]
if missing_keys:
    raise KeyError(f"Missing protocol keys {missing_keys}; available keys: {list(p)}")

targets = p["targets"]
if isinstance(targets, str):
    targets = [targets]
excluded = set(p["excluded_columns"])
group = p["group_column"]
task_metric = p["task"]["primary_metric"]
if task_metric not in ("roc_auc", "roc_auc_ovr", "roc_auc_ovr_macro"):
    raise ValueError(f"Unsupported primary_metric: {task_metric}")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def load_pkl(path):
    return pickle.loads(path.read_bytes())

# Derive feature columns from the frozen training table, excluding protocol-declared columns.
blocked = set(targets) | excluded | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in blocked]
n_train = len(train)
for c in list(feature_cols):
    s = train[c]
    if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
        if s.nunique(dropna=True) == n_train and n_train > 10:
            feature_cols.remove(c)
    elif pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=True) == n_train:
        if c.lower() in {"id", "index", "row_id", "rowid"} or c.lower().endswith("_id"):
            feature_cols.remove(c)

# Reuse the most complete saved pipeline that accepts the raw feature table.
model_paths = [
    inputs["07_model_train_logistic_QH9VENHA/model.pkl"],
    inputs["latest_model"],
    inputs["08_model_train_alternative_2548TA6F/alternative_model.pkl"],
]
objs = [load_pkl(path) for path in model_paths]
candidates = [
    o for o in objs
    if isinstance(o, Pipeline)
    and len(o.named_steps) >= 2
    and hasattr(o, "predict_proba")
    and hasattr(o, "feature_names_in_")
]
if candidates:
    pipe = max(candidates, key=lambda o: len(o.named_steps))
else:
    pre = load_pkl(inputs["05_preprocess_fit_preprocessing_5C9G845H/preprocessor.pkl"])
    sel = load_pkl(inputs["06_features_select_features_HP31G2ZF/selector.pkl"])
    mdl = load_pkl(inputs["07_model_train_logistic_QH9VENHA/model.pkl"])
    if isinstance(sel, Pipeline):
        pipe = Pipeline(list(sel.steps) + [("model", mdl)])
    else:
        pipe = Pipeline([("preprocess", pre), ("select", sel), ("model", mdl)])

if hasattr(pipe, "feature_names_in_"):
    input_cols = list(pipe.feature_names_in_)
else:
    input_cols = feature_cols

bad = set(input_cols) & blocked
if bad:
    raise ValueError(f"Pipeline input contains forbidden columns: {sorted(bad)}")
missing_cols = [c for c in input_cols if c not in val.columns]
if missing_cols:
    raise KeyError(f"Validation table is missing required columns: {missing_cols}")

X_val = val[input_cols]
y_raw = val[targets[0]]
mask = y_raw.notna()
if not mask.all():
    X_val = X_val.loc[mask]
    y_raw = y_raw.loc[mask]

classes = getattr(pipe, "classes_", None)
if classes is not None:
    mapping = {c: i for i, c in enumerate(classes)}
    try:
        y_val = np.array([mapping[v] for v in y_raw])
    except Exception:
        y_val = pd.factorize(y_raw)[0]
else:
    y_val = pd.factorize(y_raw)[0]

def scorer(estimator, X, y):
    proba = estimator.predict_proba(X)
    if proba.shape[1] == 2:
        return roc_auc_score(y, proba[:, 1])
    return roc_auc_score(y, proba, multi_class="ovr", average="macro")

res = permutation_importance(
    pipe, X_val, y_val, scoring=scorer,
    n_repeats=20, random_state=42, n_jobs=1
)

imp = pd.DataFrame({
    "feature": input_cols,
    "importance_mean": res.importances_mean,
    "importance_sd": res.importances_std,
})
imp = imp.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
for c in ["importance_mean", "importance_sd"]:
    imp[c] = imp[c].apply(lambda v: float(f"{v:.4g}"))

mlca_runtime.save_table(
    imp, "interpretation.csv",
    caption="Validation permutation importance, 20 repeats, ROC AUC"
)
top_feature = str(imp.loc[0, "feature"])
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"validation_rows={len(X_val)}")
print(f"features={len(input_cols)}")
print(f"top_feature={top_feature}")
