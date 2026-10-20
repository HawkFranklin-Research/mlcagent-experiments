import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

seed = 42
mlca_runtime.set_seed(seed)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(inputs["train"])

with open(inputs["08_model_train_logistic_Y7873DWF/model.pkl"], "rb") as f:
    base_model = pickle.load(f)

params = {"random_state": seed}
if train[target].value_counts(normalize=True).min() < 0.10:
    params["class_weight"] = "balanced"
clf = HistGradientBoostingClassifier(**params)

if isinstance(base_model, Pipeline) and hasattr(base_model, "feature_names_in_"):
    feature_cols = list(base_model.feature_names_in_)
    steps = list(base_model.steps)
    last_name = steps[-1][0]
    steps[-1] = (last_name, clf)
    model = Pipeline(steps)
else:
    with open(inputs["06_preprocess_fit_preprocessing_STWKFGVD/preprocessor.pkl"], "rb") as f:
        preprocessor = pickle.load(f)
    with open(inputs["07_features_select_features_N6F5BDYX/selector.pkl"], "rb") as f:
        selector = pickle.load(f)

    if hasattr(preprocessor, "feature_names_in_"):
        feature_cols = list(preprocessor.feature_names_in_)
    else:
        drop = set(targets + excluded + ([group_col] if group_col else []))
        feature_cols = [c for c in train.columns if c not in drop]

    def raw_input(obj):
        if not hasattr(obj, "feature_names_in_"):
            return False
        try:
            return list(obj.feature_names_in_) == feature_cols
        except Exception:
            return False

    if raw_input(selector):
        frontend = selector
    elif isinstance(selector, Pipeline) and raw_input(selector.steps[0][1]):
        frontend = selector
    elif selector is None:
        frontend = preprocessor
    else:
        if isinstance(selector, Pipeline) and any(raw_input(step) for _, step in selector.steps):
            frontend = selector
        else:
            frontend = Pipeline([("preprocessor", preprocessor), ("selector", selector)])

    steps = list(frontend.steps) if isinstance(frontend, Pipeline) else [("frontend", frontend)]
    steps.append(("classifier", clf))
    model = Pipeline(steps)

for bad in list(targets) + list(excluded) + ([group_col] if group_col else []):
    if bad in feature_cols:
        feature_cols.remove(bad)

X_train = train[feature_cols]
y_train = train[target]

mlca_runtime.fit_record(model, X_train, "train", y=y_train)
mlca_runtime.save_model(model, "alternative_model.pkl")

print(f"train rows: {len(train)}, features: {len(feature_cols)}, target: {target}")
