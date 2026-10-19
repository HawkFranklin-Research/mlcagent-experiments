import json
import pickle
import math
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    group = protocol["group_column"]
    task = protocol["task"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available keys: {list(protocol.keys())}") from e

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
target = targets[0]
drop = set(targets) | set(excluded) | ({group} if group else set())

def is_key_like(s, name):
    if s.nunique(dropna=False) != len(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        n = name.lower()
        return n in {"id", "index", "key", "code"} or n.endswith("_id") or n.startswith("id_") or n.endswith("_key") or n.startswith("key_")
    return False

candidate_cols = [c for c in train.columns if c not in drop]
feature_cols_data = [c for c in candidate_cols if not is_key_like(train[c], c)]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_feature_names(obj):
    fn = getattr(obj, "feature_names_in_", None)
    if fn is not None:
        return list(fn)
    if isinstance(obj, Pipeline):
        for _, step in obj.named_steps.items():
            fn = getattr(step, "feature_names_in_", None)
            if fn is not None:
                return list(fn)
    return None

pipeline = None
best_score = -1
for key in ["latest_model", "07_model_train_logistic_H2ZAD1XK/model.pkl", "08_model_train_alternative_XPWMPVN8/alternative_model.pkl"]:
    path = inputs.get(key)
    if path is None or not path.exists():
        continue
    obj = load_pkl(path)
    fn = get_feature_names(obj)
    if fn is not None and set(fn) == set(feature_cols_data):
        score = len(getattr(obj, "named_steps", {}))
        if score > best_score:
            pipeline, best_score = obj, score

if pipeline is None:
    pp = load_pkl(inputs["05_preprocess_fit_preprocessing_WBRZJAT5/preprocessor.pkl"])
    sel = load_pkl(inputs["06_features_select_features_PDWSHJGW/selector.pkl"])
    mdl = load_pkl(inputs["07_model_train_logistic_H2ZAD1XK/model.pkl"])
    mdl_fn = get_feature_names(mdl)
    if mdl_fn is not None and set(mdl_fn) == set(feature_cols_data):
        pipeline = mdl
    else:
        steps = []
        if isinstance(sel, Pipeline) and "preprocessor" in getattr(sel, "named_steps", {}):
            steps.append(("frontend", sel))
        elif isinstance(pp, Pipeline) and "selector" in getattr(pp, "named_steps", {}):
            steps.append(("frontend", pp))
        else:
            steps.append(("preprocessor", pp))
            steps.append(("selector", sel))
        steps.append(("model", mdl))
        pipeline = Pipeline(steps)

fn = get_feature_names(pipeline)
feature_cols = list(fn) if fn is not None and set(fn) == set(feature_cols_data) else feature_cols_data

def get_classes(obj):
    if hasattr(obj, "classes_"):
        return list(obj.classes_)
    if isinstance(obj, Pipeline):
        for _, step in reversed(obj.named_steps.items()):
            if hasattr(step, "classes_"):
                return list(step.classes_)
    return None

y_raw = val[target].copy()
classes = get_classes(pipeline)
if classes is not None and len(classes) == 2:
    y_vals = set(y_raw.dropna().unique())
    if not y_vals.issubset(set(classes)):
        train_labels = sorted(train[target].dropna().unique())
        if len(train_labels) == len(classes):
            mapping = dict(zip(train_labels, classes))
            y_eval = y_raw.map(mapping)
            if y_eval.isna().any():
                raise ValueError("Validation target could not be mapped to model classes.")
        else:
            y_eval = y_raw
    else:
        y_eval = y_raw
else:
    y_eval = y_raw

mask = y_eval.notna()
X_val = val.loc[mask, feature_cols].copy()
y_eval = y_eval.loc[mask]

metric = task["primary_metric"]
n_classes = task.get("n_classes", 2)
if n_classes == 2 or metric == "roc_auc":
    scoring = "roc_auc"
else:
    scoring = "roc_auc_ovr"

result = permutation_importance(
    pipeline,
    X_val,
    y_eval,
    scoring=scoring,
    n_repeats=20,
    random_state=42,
    n_jobs=1,
)

def sig4(x):
    if pd.isna(x) or x == 0:
        return float(x) if not pd.isna(x) else x
    return round(float(x), 3 - int(math.floor(math.log10(abs(float(x))))))

imp = pd.DataFrame({
    "feature": X_val.columns,
    "importance_mean": result.importances_mean,
    "importance_sd": result.importances_std,
})
imp = imp.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
for c in ["importance_mean", "importance_sd"]:
    imp[c] = imp[c].map(sig4)

mlca_runtime.save_table(imp, "interpretation.csv", caption="Permutation importance on validation rows, 20 repeats, ROC AUC.")
top_feature = str(imp.iloc[0]["feature"])
top_mean = float(imp.iloc[0]["importance_mean"])
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"validation rows: {len(X_val)}, features: {len(feature_cols)}")
print(f"pipeline: {type(pipeline).__name__}, steps: {list(getattr(pipeline, 'named_steps', {}).keys())}")
print(f"scoring: {scoring}, repeats: 20")
print(f"top feature: {top_feature} mean={top_mean:.4g}")
