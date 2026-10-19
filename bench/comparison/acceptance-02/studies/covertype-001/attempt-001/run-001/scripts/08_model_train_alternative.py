import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

mlca_runtime.set_seed(42)
SEED = 42

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
feat_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
X_train = train[feat_cols]
y_train = train[targets[0]]

with open(inputs["07_model_train_logistic_Z4Z606HV/model.pkl"], "rb") as f:
    old_model = pickle.load(f)

if not isinstance(old_model, Pipeline):
    raise RuntimeError("expected saved model to be a Pipeline")

steps = list(old_model.steps[:-1]) + [
    ("model", HistGradientBoostingClassifier(random_state=SEED))
]
pipe = Pipeline(steps)

fitted = mlca_runtime.fit_record(pipe, X_train, "train", y_train)
mlca_runtime.save_model(fitted, "alternative_model.pkl")

print(f"train_rows={len(X_train)} features={len(feat_cols)} "
      f"classes={len(fitted.named_steps['model'].classes_)}")
