import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

with open(inputs["05_preprocess_fit_preprocessing_BKKYG4P6/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_GQKA7M4V/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def is_keylike(s):
    if s.dtype == object:
        return s.nunique(dropna=True) == len(s) and len(s) > 0
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == len(s)
    return False

if hasattr(pre, "feature_names_in_"):
    feature_cols = [c for c in pre.feature_names_in_ if c not in targets and c not in excluded and c != group]
else:
    feature_cols = [c for c in train.columns
                    if c not in targets and c not in excluded and c != group
                    and not is_keylike(train[c])]

X_train = train[feature_cols]
y_train = train[targets[0]]

def contains(pipe, obj):
    if not isinstance(pipe, Pipeline):
        return False
    return any(step is obj for _, step in pipe.named_steps.items())

frontend = sel if contains(sel, pre) else Pipeline([("pre", pre), ("sel", sel)])

counts = y_train.value_counts()
cw = "balanced" if counts.min() / counts.sum() < 0.10 else None

clf = HistGradientBoostingClassifier(random_state=42, class_weight=cw)
model = Pipeline([("frontend", frontend), ("clf", clf)])

mlca_runtime.fit_record(model, X_train, "train", y_train)
mlca_runtime.save_model(model, "alternative_model.pkl")

print("features:", len(feature_cols), "train rows:", len(X_train))
print("class_weight:", cw)
print("saved alternative_model.pkl")
