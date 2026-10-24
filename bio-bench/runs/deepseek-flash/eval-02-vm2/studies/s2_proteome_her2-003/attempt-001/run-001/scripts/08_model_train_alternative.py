import mlca_runtime
import pickle
import json
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])
target_col = protocol["targets"][0]
y_train = train[target_col]

with open(inputs["07_model_train_logistic_4PCHXMGW/model.pkl"], "rb") as f:
    model = pickle.load(f)

if not isinstance(model, Pipeline):
    raise TypeError("Loaded model is not a sklearn Pipeline")

try:
    feature_cols = model.feature_names_in_
except AttributeError:
    feature_cols = model.steps[0][1].feature_names_in_

X_train = train[feature_cols]

new_pipeline = clone(model)
last_step = new_pipeline.steps[-1][0]
new_pipeline.set_params(**{last_step: HistGradientBoostingClassifier(random_state=42)})
for name, step in new_pipeline.named_steps.items():
    if hasattr(step, 'set_output'):
        step.set_output(transform='pandas')

mlca_runtime.fit_record(new_pipeline, X_train, "train", y_train)
mlca_runtime.save_model(new_pipeline, "alternative_model.pkl")
print(f"Trained HistGradientBoostingClassifier on {X_train.shape[0]} rows, {X_train.shape[1]} features")
print("Saved alternative_model.pkl")
