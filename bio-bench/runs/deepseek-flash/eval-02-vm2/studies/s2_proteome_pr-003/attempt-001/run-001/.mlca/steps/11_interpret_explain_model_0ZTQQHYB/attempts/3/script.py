import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

SEED = 42
mlca_runtime.set_seed(SEED)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key '{key}'; available: {list(protocol.keys())}")
targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group_column = protocol.get("group_column")
task = protocol["task"]
n_classes = int(task.get("n_classes", 2))
metric = task.get("primary_metric", "roc_auc")
if len(targets) != 1:
    raise ValueError(f"Expected exactly one target, got {targets}")
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

with open(inputs["05_preprocess_fit_preprocessing_YCEBPQHT/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_FXHKDZPP/selector.pkl"], "rb") as f:
    selector = pickle.load(f)
with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)


def unwrap(o):
    if isinstance(o, dict):
        for key in ("model", "pipeline", "estimator", "best_model"):
            if key in o:
                return unwrap(o[key])
    return o


model = unwrap(model)


def input_names(obj):
    obj = unwrap(obj)
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_") and obj.feature_names_in_ is not None:
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline) and obj.steps:
        return input_names(obj.steps[0][1])
    return None


if isinstance(model, Pipeline):
    predictor = model
else:
    stages = []
    if pre is not None:
        stages.append(("preprocessor", pre))
    if selector is not None:
        stages.append(("selector", selector))
    stages.append(("model", model))
    predictor = Pipeline(stages)

drop = set(targets) | set(excluded)
if group_column:
    drop.add(group_column)
eligible = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)) and s.nunique(dropna=False) == len(s):
        continue
    eligible.append(c)

raw_schema = input_names(predictor)
if raw_schema is None:
    raw_schema = eligible
bad = set(raw_schema) & drop
if bad:
    raise ValueError(f"Model input schema contains forbidden columns: {sorted(bad)}")
missing_cols = [c for c in raw_schema if c not in val.columns]
if missing_cols:
    raise KeyError(f"Validation missing columns: {missing_cols[:5]}")

n_features = len(raw_schema)

if n_features > 500:
    final = predictor.steps[-1][1] if isinstance(predictor, Pipeline) else predictor
    coef = getattr(final, "coef_", None)
    fi = getattr(final, "feature_importances_", None)
    if coef is not None:
        arr = np.abs(np.asarray(coef))
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        method = "coefficients"
    elif fi is not None:
        arr = np.asarray(fi)
        method = "feature_importances"
    else:
        raise RuntimeError("More than 500 input features and no coefficients/feature_importances_; cannot rank reliably.")
    names = None
    if isinstance(predictor, Pipeline):
        try:
            names = list(predictor[:-1].get_feature_names_out(input_features=raw_schema))
        except Exception:
            names = None
    if names is None or len(names) != len(arr):
        names = raw_schema if len(raw_schema) == len(arr) else [f"feature_{i}" for i in range(len(arr))]
    out = pd.DataFrame({"feature": names, "importance_mean": arr, "importance_sd": np.nan})
    n_repeats = 0
else:
    X_val = val[raw_schema].copy()
    classes = sorted(pd.unique(val[target].dropna()))
    mapping = {c: i for i, c in enumerate(classes)}
    y_val = val[target].map(mapping).to_numpy()

    def score(df):
        proba = predictor.predict_proba(df)
        if n_classes > 2:
            return roc_auc_score(y_val, proba, multi_class="ovr", average="macro")
        return roc_auc_score(y_val, proba[:, 1])

    if metric != "roc_auc":
        raise ValueError(f"Unsupported primary_metric for permutation scoring: {metric}")
    n_repeats = 20
    rng = np.random.RandomState(SEED)
    baseline = score(X_val)
    cols = list(raw_schema)
    imp = np.zeros((len(cols), n_repeats))
    for j, col in enumerate(cols):
        orig = X_val[col].to_numpy(copy=True)
        for r in range(n_repeats):
            X_val[col] = rng.permutation(orig)
            imp[j, r] = baseline - score(X_val)
        X_val[col] = orig
    out = pd.DataFrame({
        "feature": cols,
        "importance_mean": imp.mean(axis=1),
        "importance_sd": imp.std(axis=1, ddof=0),
    })
    method = "permutation"

out = out.sort_values("importance_mean", ascending=False).head(15).copy()
out["importance_mean"] = pd.to_numeric(out["importance_mean"], errors="coerce").round(6)
out["importance_sd"] = pd.to_numeric(out["importance_sd"], errors="coerce").round(6)
mlca_runtime.save_table(out, "interpretation.csv", caption="Frozen model feature importance")

top_name = str(out.iloc[0]["feature"]) if len(out) else ""
if top_name:
    mlca_runtime.record_fact_candidate("top_feature", top_name, source="interpretation.csv")

print(f"val rows={len(val)}; input features={n_features}; method={method}; repeats={n_repeats}")
print(f"top_feature={top_name}")
