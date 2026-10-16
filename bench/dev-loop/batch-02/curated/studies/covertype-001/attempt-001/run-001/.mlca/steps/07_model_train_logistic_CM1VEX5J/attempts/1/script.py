import mlca_runtime
import json, pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from pandas.api.types import is_object_dtype, is_string_dtype, is_integer_dtype

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)

train = pd.read_parquet(inputs["train"])

with open(inputs["05_preprocess_fit_preprocessing_R1SQG5ET/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_select_features_830E7YS6/selector.pkl"], "rb") as f:
    selector = pickle.load(f)
with open(inputs["05_preprocess_fit_preprocessing_R1SQG5ET/manifest.json"]) as f:
    prep_manifest = json.load(f)
with open(inputs["06_features_select_features_830E7YS6/manifest.json"]) as f:
    sel_manifest = json.load(f)

def get_input_columns(obj, manifest):
    if hasattr(obj, "feature_names_in_") and obj.feature_names_in_ is not None:
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        first_name = list(obj.named_steps.keys())[0]
        first = obj.named_steps[first_name]
        if hasattr(first, "feature_names_in_") and first.feature_names_in_ is not None:
            return list(first.feature_names_in_)
    for key in ["input_columns", "input_features", "features", "raw_features"]:
        if manifest and key in manifest:
            return list(manifest[key])
    return None

selector_includes_prep = False
if isinstance(selector, Pipeline):
    for name, step in selector.named_steps.items():
        if isinstance(step, ColumnTransformer) or name.lower() in ("preprocessing", "preprocessor"):
            selector_includes_prep = True
            break

if selector_includes_prep:
    cols = get_input_columns(selector, sel_manifest)
    steps = [("frontend", selector)]
else:
    cols = get_input_columns(preprocessor, prep_manifest)
    steps = [("preprocessing", preprocessor), ("selector", selector)]

if cols is None:
    cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
    drop = []
    for c in cols:
        s = train[c]
        if is_object_dtype(s) or is_string_dtype(s):
            if s.nunique() == len(train):
                drop.append(c)
        elif is_integer_dtype(s):
            if s.nunique() == len(train) and any(k in c.lower() for k in ["id", "key", "index"]):
                drop.append(c)
    cols = [c for c in cols if c not in drop]

cols = [c for c in cols if c not in targets and c not in excluded and c != group]

model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
steps.append(("model", model))
pipeline = Pipeline(steps)

X_train = train[cols]
y_train = train[targets[0]]
mlca_runtime.fit_record(pipeline, X_train, "train", y_train)
mlca_runtime.save_model(pipeline, "model.pkl")

print(f"features={len(cols)} rows={len(X_train)}")
print(f"selector_includes_prep={selector_includes_prep}")
print(f"model={type(model).__name__}")
