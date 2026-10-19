import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
try:
    targets = list(protocol["targets"])
    excluded = list(protocol["excluded_columns"])
    task = protocol["task"]
except KeyError as e:
    raise KeyError(f"Missing {e}; available keys: {list(protocol.keys())}") from e
group = protocol.get("group_column")
metric = task.get("primary_metric") or "roc_auc"
n_classes = int(task.get("n_classes", 2))

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
target = targets[0]
val = val[val[target].notna()]

def key_like(s):
    return (s.nunique(dropna=False) == len(s) and
            (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or
             isinstance(s.dtype, pd.CategoricalDtype)))

features = [c for c in train.columns
            if c not in targets and c not in excluded and c != group and not key_like(train[c])]
missing = [c for c in features if c not in val.columns]
if missing:
    raise ValueError(f"Missing feature columns in val: {missing}")
X_raw = val[features]
print(f"train rows={len(train)}, val rows={len(val)}, features={len(features)}")

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inputs["05_preprocess_fit_preprocessing_F9B0NJ26/preprocessor.pkl"])
sel = load_pkl(inputs["06_features_select_features_AA6KHSXZ/selector.pkl"])
model = load_pkl(inputs["07_model_train_logistic_23SCN2HD/model.pkl"])
alt = load_pkl(inputs["08_model_train_alternative_9VT6FVM1/alternative_model.pkl"])
latest = load_pkl(inputs["latest_model"])

def cols_of(obj):
    return list(getattr(obj, "feature_names_in_", []))

def works(obj, X):
    if not hasattr(obj, "predict_proba"):
        return False
    try:
        cols = cols_of(obj)
        Xs = X[cols] if cols else X
        obj.predict_proba(Xs.iloc[:1])
        return True
    except Exception:
        return False

pipeline = None
for cand in (latest, model, alt):
    if works(cand, X_raw):
        pipeline = cand
        break
if pipeline is None:
    options = ([pre, sel, model], [pre, model], [sel, model],
               [pre, sel, alt], [pre, alt], [sel, alt])
    for steps in options:
        p = Pipeline([(f"s{i}", s) for i, s in enumerate(steps)])
        if works(p, X_raw):
            pipeline = p
            break
if pipeline is None:
    raise RuntimeError("No usable frozen pipeline found")

def get_classes(obj):
    for o in [obj] + list(getattr(obj, "named_steps", {}).values())[::-1]:
        c = getattr(o, "classes_", None)
        if c is not None:
            return c
    return None

cols = cols_of(pipeline) or features
bad = set(cols) & (set(targets) | set(excluded) | ({group} if group else set()))
if bad:
    raise ValueError(f"Non-feature columns in pipeline input: {bad}")
X_val = val[cols]

cls = get_classes(pipeline)
y_val = val[target]
if n_classes == 2 and metric in ("roc_auc", "auc") and cls is not None and len(cls) == 2:
    y_score = (y_val == cls[1]).astype(int)
else:
    y_score = y_val
scoring = ("roc_auc" if n_classes == 2 else "roc_auc_ovr") if metric in ("roc_auc", "auc") else metric

res = permutation_importance(
    pipeline, X_val, y_score, scoring=scoring,
    n_repeats=20, random_state=42, n_jobs=1
)
imp = pd.DataFrame({
    "feature": X_val.columns,
    "importance_mean": res.importances_mean,
    "importance_sd": res.importances_std,
}).sort_values("importance_mean", ascending=False).head(15)
for c in ("importance_mean", "importance_sd"):
    imp[c] = imp[c].map(lambda v: float(f"{v:.4g}"))

mlca_runtime.save_table(imp, "interpretation.csv",
                        caption="Permutation importance on validation rows")
top = str(imp.iloc[0]["feature"])
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
print(f"top_feature={top}")
