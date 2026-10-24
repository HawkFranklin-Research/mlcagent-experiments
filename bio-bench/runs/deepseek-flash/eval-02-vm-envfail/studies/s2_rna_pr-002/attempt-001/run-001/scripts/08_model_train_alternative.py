import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
excluded = set(protocol.get("excluded_columns", []))
group_col = protocol.get("group_column")
task = protocol["task"]
primary = task["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])


def key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == len(s) and len(s) > 1
    return False


feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group_col and not key_like(train[c])
]

with open(inp["07_model_train_logistic_RTDQBRJK/model.pkl"], "rb") as f:
    base = pickle.load(f)
with open(inp["06_features_select_features_8X366KXZ/selector.pkl"], "rb") as f:
    selector = pickle.load(f)


def has_selector(p):
    return isinstance(p, Pipeline) and any(
        "select" in name.lower() or "Select" in type(p.named_steps[name]).__name__
        for name in p.named_steps
    )


if isinstance(base, Pipeline) and has_selector(base):
    front_steps = list(base.steps[:-1])
elif isinstance(selector, Pipeline) and len(selector.steps) >= 2:
    front_steps = list(selector.steps)
    last_name = type(front_steps[-1][1]).__name__ if front_steps else ""
    if "Classifier" in last_name or "Regressor" in last_name:
        front_steps = front_steps[:-1]
else:
    with open(inp["05_preprocess_fit_preprocessing_WM2D925J/preprocessor.pkl"], "rb") as f:
        pre = pickle.load(f)
    pre_steps = list(pre.steps) if isinstance(pre, Pipeline) else [("preprocessor", pre)]
    front_steps = pre_steps + [("selector", selector)]

fit_cols = None
for obj in (base, selector, front_steps[0][1] if front_steps else None):
    if obj is not None and hasattr(obj, "feature_names_in_"):
        fit_cols = list(obj.feature_names_in_)
        break
if fit_cols is None:
    fit_cols = feature_cols

forbidden = set(targets) | excluded | ({group_col} if group_col else set())
bad = sorted(set(fit_cols) & forbidden)
if bad:
    raise ValueError(f"feature schema contains forbidden columns: {bad}")

target = targets[0]
X_train = train[fit_cols]
y_train = train[target]

classifier = HistGradientBoostingClassifier(random_state=42)
pipeline = Pipeline(front_steps + [("classifier", classifier)])
fitted = mlca_runtime.fit_record(pipeline, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "alternative_model.pkl")

if primary == "roc_auc" and target in val.columns:
    proba = fitted.predict_proba(val[fit_cols])
    classes = list(fitted.classes_)
    if len(classes) == 2:
        y_bin = (val[target] == classes[1]).astype(int)
        if y_bin.nunique() == 2:
            auc = roc_auc_score(y_bin, proba[:, 1])
            mlca_runtime.record_metric(primary, float(auc), "val")
            print(f"val {primary}={auc:.4f}")

print(f"train rows={len(train)} features={len(fit_cols)}")
