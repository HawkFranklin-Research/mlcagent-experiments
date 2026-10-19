import mlca_runtime, json, pickle
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"missing {key}; available={list(protocol)}")
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
task = protocol["task"]
primary = task.get("primary_metric", "roc_auc_ovr")
n_classes = task.get("n_classes", 2)
seed = 42

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
print(f"train rows={len(train)} val rows={len(val)}")

target = targets[0]

def key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=False) == len(s) and not pd.api.types.is_numeric_dtype(s)
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False

drop = set(targets + excluded + ([group] if group else []))
features = [c for c in train.columns if c not in drop and not key_like(train[c])]

def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model = None
for key in [
    "latest_model",
    "08_model_train_alternative_5HD55PJ2/alternative_model.pkl",
    "07_model_train_logistic_H5RFG7PA/model.pkl",
]:
    try:
        obj = load(inp[key])
    except Exception:
        continue
    if hasattr(obj, "predict_proba"):
        model = obj
        break
if model is None:
    raise RuntimeError("no saved model with predict_proba")

if not isinstance(model, Pipeline):
    pre = load(inp["05_preprocess_fit_preprocessing_8APC2384/preprocessor.pkl"])
    sel = load(inp["06_features_select_features_23ZRTGDS/selector.pkl"])
    steps = []
    if isinstance(pre, Pipeline) and any("select" in k.lower() for k in pre.named_steps):
        steps = [("preprocessor", pre)]
    else:
        steps = [("preprocessor", pre), ("selector", sel)]
    steps.append(("model", model))
    model = Pipeline(steps)

def input_names(est):
    names = getattr(est, "feature_names_in_", None)
    if names is None and hasattr(est, "named_steps"):
        for step in est.named_steps.values():
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                break
    return list(names) if names is not None else None

names = input_names(model)
if names is not None:
    if any(c in targets or c in excluded for c in names):
        raise ValueError("model input contains target/excluded columns")
    features = names

missing = [c for c in features if c not in val.columns]
if missing:
    raise KeyError(f"missing val columns: {missing}")

X = val[features]
y = val[target]
n_classes = y.nunique() if n_classes is None else n_classes
if primary in ("roc_auc_ovr", "roc_auc_ovo"):
    scoring = primary
elif primary == "roc_auc" and n_classes > 2:
    scoring = "roc_auc_ovr"
else:
    scoring = "roc_auc_ovr" if n_classes > 2 else "roc_auc"

res = permutation_importance(
    model, X, y, scoring=scoring, n_repeats=20, random_state=seed, n_jobs=-1
)
out = pd.DataFrame({
    "feature": features,
    "importance_mean": res.importances_mean,
    "importance_sd": res.importances_std,
})
out = out.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
for col in ("importance_mean", "importance_sd"):
    out[col] = out[col].round(4)

mlca_runtime.save_table(out, "interpretation.csv", caption="Permutation importance on validation rows")
top = str(out.iloc[0]["feature"]) if len(out) else ""
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
print(f"scoring={scoring} top_feature={top}")
