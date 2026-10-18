import json, pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns", "group_column"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]

train = pd.read_parquet(inputs["train"])
for t in targets:
    if t not in train.columns:
        raise KeyError(f"Target {t!r} not in training columns")

def is_key_like(s: pd.Series) -> bool:
    if len(s) == 0 or s.nunique(dropna=False) != len(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s):
        return True
    return False

excluded_set = set(targets) | set(excluded)
if group is not None:
    excluded_set.add(group)
features = [c for c in train.columns if c not in excluded_set and not is_key_like(train[c])]
X_train = train[features]
y_train = train[targets].astype(float)

def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

selector = load(inputs["06_features_retain_features_JTARV5X3/selector_all.pkl"])
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

if isinstance(selector, Pipeline):
    steps = list(selector.steps)
    if steps and steps[-1][0] in ("model", "regressor", "classifier", "estimator"):
        steps[-1] = ("model", rf)
    else:
        steps.append(("model", rf))
    pipeline = Pipeline(steps)
else:
    pre = load(inputs["05_preprocess_fit_preprocessing_0CQ66EDA/preprocessor.pkl"])
    steps = list(pre.steps) if isinstance(pre, Pipeline) else [("preprocessor", pre)]
    steps.extend([("selector", selector), ("model", rf)])
    pipeline = Pipeline(steps)

for _, step in pipeline.steps:
    if hasattr(step, "set_output"):
        try:
            step.set_output(transform="pandas")
        except Exception:
            pass

fitted = mlca_runtime.fit_record(pipeline, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "alternative_model.pkl")
print(f"train rows: {len(train)}")
print(f"features: {len(features)}")
print(f"targets: {targets}")
