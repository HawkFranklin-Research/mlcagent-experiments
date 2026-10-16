import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

seed = 42
mlca_runtime.set_seed(seed)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns") or []
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = [c for c in train.columns if c not in drop]
X_train = train[feature_cols]
y_train = train[targets[0]]

with open(inputs["05_preprocess_fit_preprocessing_R1SQG5ET/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_830E7YS6/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def steps_of(obj, name):
    return list(obj.steps) if isinstance(obj, Pipeline) else [(name, obj)]

def accepts_raw(obj, raw):
    if hasattr(obj, "feature_names_in_"):
        return set(obj.feature_names_in_).issubset(raw)
    if isinstance(obj, Pipeline):
        return accepts_raw(list(obj.named_steps.values())[0], raw)
    return False

raw = set(feature_cols)
if accepts_raw(sel, raw):
    steps = steps_of(sel, "selector")
else:
    steps = steps_of(pre, "preprocessor") + steps_of(sel, "selector")

steps.append(("clf", HistGradientBoostingClassifier(random_state=seed)))
pipe = Pipeline(steps)

mlca_runtime.fit_record(pipe, X_train, "train", y_train)
mlca_runtime.save_model(pipe, "alternative_model.pkl")
print("saved alternative_model.pkl", len(feature_cols), "features")
