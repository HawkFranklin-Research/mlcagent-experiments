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

target = protocol["targets"][0]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])

with open(inputs["07_model_train_logistic_GZE7Y5E0/model.pkl"], "rb") as f:
    model = pickle.load(f)

if not isinstance(model, Pipeline):
    with open(inputs["05_preprocess_fit_preprocessing_3K40DCMA/preprocessor.pkl"], "rb") as f:
        pre = pickle.load(f)
    with open(inputs["06_features_select_features_XBX6AN5A/selector.pkl"], "rb") as f:
        sel = pickle.load(f)
    steps = []
    for obj in (pre, sel):
        if isinstance(obj, Pipeline):
            steps.extend(obj.steps)
        else:
            steps.append((f"step_{len(steps)}", obj))
    steps.append(("classifier", HistGradientBoostingClassifier(random_state=seed)))
    pipeline = Pipeline(steps)
    first = pre.steps[0][1] if isinstance(pre, Pipeline) else pre
    feature_cols = list(first.feature_names_in_) if hasattr(first, "feature_names_in_") else [
        c for c in train.columns if c not in protocol["targets"] and c not in excluded and c != group
    ]
else:
    steps = list(model.steps)
    last_name, last_step = steps[-1]
    if hasattr(last_step, "predict"):
        new_steps = steps[:-1] + [(last_name, HistGradientBoostingClassifier(random_state=seed))]
    else:
        new_steps = steps + [("classifier", HistGradientBoostingClassifier(random_state=seed))]
    pipeline = Pipeline(new_steps)
    first_step = steps[0][1]
    if hasattr(first_step, "feature_names_in_"):
        feature_cols = list(first_step.feature_names_in_)
    elif hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    else:
        feature_cols = [c for c in train.columns if c not in protocol["targets"] and c not in excluded and c != group]

feature_cols = [c for c in feature_cols if c not in protocol["targets"] and c not in excluded]

y_train = train[target]
min_frac = y_train.value_counts(normalize=True).min()
if min_frac < 0.10:
    last_step_name = list(pipeline.named_steps)[-1]
    pipeline.named_steps[last_step_name].set_params(class_weight="balanced")

mlca_runtime.fit_record(pipeline, train[feature_cols], partition="train", y=y_train)
mlca_runtime.save_model(pipeline, "alternative_model.pkl")

print(f"Train rows: {len(train)}, features: {len(feature_cols)}")
print("Saved alternative_model.pkl")
