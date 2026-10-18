import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
gcol = protocol.get("group_column")
primary = protocol["task"]["primary_metric"]
if len(targets) != 1:
    raise ValueError("Expected exactly one target column")
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return len(s) > 0 and s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        if len(s) == 0 or s.nunique(dropna=False) != len(s):
            return False
        name = str(s.name).lower()
        if any(k in name for k in ("id", "key", "index", "code", "row", "seq")):
            return True
        v = np.sort(s.dropna().to_numpy())
        return np.array_equal(v, np.arange(len(v))) or np.array_equal(v, np.arange(1, len(v) + 1))
    return False

feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != gcol and not key_like(train[c])
]
val_features = val[feature_cols].copy()

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inputs["05_preprocess_fit_preprocessing_DYJSGEW8/preprocessor.pkl"])
sel = load_pkl(inputs["06_features_select_features_KAD36QCW/selector.pkl"])

def first_step(obj):
    if isinstance(obj, Pipeline):
        return next(iter(obj.named_steps.values()))
    return obj

def accepts_raw(obj):
    first = first_step(obj)
    if hasattr(first, "feature_names_in_"):
        names = set(first.feature_names_in_)
        return not set(feature_cols).isdisjoint(names)
    if hasattr(first, "transformers_") or hasattr(first, "named_transformers_"):
        return True
    if isinstance(first, Pipeline):
        return True
    return False

def build_frontend(sel_obj):
    first = first_step(sel_obj)
    if hasattr(first, "feature_names_in_"):
        if not set(feature_cols).isdisjoint(set(first.feature_names_in_)):
            return sel_obj
    if hasattr(first, "transformers_") or hasattr(first, "named_transformers_"):
        return sel_obj
    if isinstance(sel_obj, Pipeline):
        return Pipeline([("preprocessor", pre), ("selector", sel_obj)])
    return Pipeline([("preprocessor", pre), ("selector", sel_obj)])

def build_predictor(model):
    if accepts_raw(model):
        return model
    if isinstance(model, Pipeline):
        return Pipeline([("preprocessor", pre), ("model", model)])
    return Pipeline([("frontend", build_frontend(sel)), ("model", model)])

def predictor_input(pred):
    first = first_step(pred)
    expected = getattr(pred, "feature_names_in_", None)
    if expected is None:
        expected = getattr(first, "feature_names_in_", None)
    if expected is not None and all(c in val.columns for c in expected):
        return val[list(expected)]
    return val_features

def final_est(pred):
    return pred.named_steps[list(pred.named_steps)[-1]] if isinstance(pred, Pipeline) else pred

candidates = {
    "logistic": inputs["07_model_train_logistic_KTFJXM94/model.pkl"],
    "alternative": inputs["08_model_train_alternative_M33R6NHM/alternative_model.pkl"],
    "latest": inputs["latest_model"],
}
seen, rows, scores = set(), [], {}
for name, path in candidates.items():
    rp = str(path.resolve())
    if rp in seen:
        continue
    seen.add(rp)
    model = load_pkl(path)
    pred = build_predictor(model)
    X = predictor_input(pred)
    proba = pred.predict_proba(X)
    est = final_est(pred)
    classes = getattr(est, "classes_", None)
    if classes is None:
        classes = np.sort(val[target].unique())
    y_val = val[target]
    auc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)
    pred_labels = classes[np.argmax(proba, axis=1)]
    f1 = f1_score(y_val, pred_labels, average="macro", labels=classes, zero_division=0)
    scores[name] = auc
    rows.append({"model": name, "metric": primary, "value": float(f"{auc:.4g}"), "threshold": np.nan})
    rows.append({"model": name, "metric": "macro_f1", "value": float(f"{f1:.4g}"), "threshold": np.nan})
    mlca_runtime.record_metric(primary, float(auc), "val")

metrics_df = pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"])
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics by model")
best_name = max(scores, key=scores.get)
best_val = scores[best_name]
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", float(best_val), source="validation_metrics.csv")
print("features", len(feature_cols), "val rows", len(val))
print("best", best_name, primary, round(float(best_val), 4))
