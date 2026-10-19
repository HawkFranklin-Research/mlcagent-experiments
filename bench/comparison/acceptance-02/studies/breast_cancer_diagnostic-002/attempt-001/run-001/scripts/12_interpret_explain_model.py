import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import mlca_runtime

SEED = 42
mlca_runtime.set_seed(SEED)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol)}")
if "primary_metric" not in protocol["task"]:
    raise KeyError(f"Missing task.primary_metric; task keys: {list(protocol['task'])}")

targets = protocol["targets"]
if isinstance(targets, str):
    targets = [targets]
target = targets[0]
excluded = protocol["excluded_columns"] or []
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def key_like(s):
    if len(s) == 0 or s.nunique(dropna=False) != len(s):
        return False
    return s.dtype.kind in "OUS" or s.dtype.kind in "iu"

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
allowed = [
    c for c in train.columns
    if c not in drop and not key_like(train[c])
]

def complete_pipeline(obj):
    if not isinstance(obj, Pipeline) or not hasattr(obj, "predict_proba"):
        return False
    cols = getattr(obj, "feature_names_in_", None)
    if cols is None or len(cols) == 0:
        return False
    return all(c in allowed for c in cols)

pipe = None
for key in [
    "latest_model",
    "09_model_train_alternative_QMPABASH/alternative_model.pkl",
    "08_model_train_logistic_71AKM31B/model.pkl",
]:
    path = inputs.get(key)
    if path is None or not path.exists():
        continue
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if complete_pipeline(obj):
        pipe = obj
        break
if pipe is None:
    raise ValueError(f"No complete frozen pipeline accepting raw features; inputs={list(inputs)}")

feature_cols = list(pipe.feature_names_in_)
missing = [c for c in feature_cols if c not in val.columns]
if missing:
    raise KeyError(f"Validation missing pipeline features: {missing}")
X_val = val.loc[:, feature_cols]
y_val = val[target]

def encode_binary_y(y, estimator):
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        classes = getattr(estimator.steps[-1][1], "classes_", None)
    y = pd.Series(y).copy()
    if classes is not None and len(classes) == 2:
        classes = list(classes)
        uniq = set(pd.unique(y.dropna()))
        if uniq.issubset(set(classes)):
            return y.map({c: i for i, c in enumerate(classes)})
    uniq_sorted = sorted(pd.unique(y.dropna()))
    return y.map({c: i for i, c in enumerate(uniq_sorted)})

task = protocol["task"]
n_classes = task.get("n_classes", y_val.nunique())
if n_classes == 2 or task.get("type") == "binary":
    scoring = "roc_auc"
    y_eval = encode_binary_y(y_val, pipe)
else:
    scoring = "roc_auc_ovr"
    y_eval = y_val

result = permutation_importance(
    pipe,
    X_val,
    y_eval,
    scoring=scoring,
    n_repeats=20,
    random_state=SEED,
    n_jobs=1,
)

imp = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": result.importances_mean,
    "importance_sd": result.importances_std,
})
imp = imp.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
for col in ("importance_mean", "importance_sd"):
    imp[col] = imp[col].map(lambda v: float(f"{v:.4g}"))

mlca_runtime.save_table(
    imp,
    "interpretation.csv",
    caption="Validation permutation importance, 20 repeats, compatible ROC AUC scoring.",
)
top_feature = str(imp.iloc[0]["feature"]) if len(imp) else ""
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"train rows={len(train)} val rows={len(val)}")
print(f"pipeline steps={len(pipe.named_steps)} features={len(feature_cols)}")
print(f"top feature={top_feature}")
