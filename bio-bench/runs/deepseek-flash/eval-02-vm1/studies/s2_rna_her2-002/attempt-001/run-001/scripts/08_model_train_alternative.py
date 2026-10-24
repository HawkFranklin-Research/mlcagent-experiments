import json
import pickle

import mlca_runtime
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)

inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]

train = pd.read_parquet(inp["train"])
with open(inp["07_model_train_logistic_NWPYRQ9N/model.pkl"], "rb") as f:
    base = pickle.load(f)


def raw_columns(obj):
    if hasattr(obj, "feature_names_in_") and obj.feature_names_in_ is not None:
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, "feature_names_in_") and step.feature_names_in_ is not None:
                return list(step.feature_names_in_)
    raise ValueError("Could not determine raw feature schema from saved model")


cols = raw_columns(base)
excluded = set(protocol.get("excluded_columns", []))
if target in cols or any(c in excluded for c in cols):
    raise ValueError("Saved feature schema includes target or excluded columns")

X = train[cols].copy()
y = train[target].copy()

hgb = HistGradientBoostingClassifier(random_state=42)

if isinstance(base, Pipeline):
    steps = list(base.steps[:-1]) + [("classifier", hgb)]
    alt = Pipeline(steps)
else:
    with open(inp["05_preprocess_fit_preprocessing_R1GAM795/preprocessor.pkl"], "rb") as f:
        pre = pickle.load(f)
    with open(inp["06_features_select_features_FJMHG7YC/selector.pkl"], "rb") as f:
        sel = pickle.load(f)
    if isinstance(sel, Pipeline) and any(
        n in sel.named_steps for n in ("preprocessor", "preprocess")
    ):
        steps = list(sel.steps[:-1]) + [("classifier", hgb)]
    else:
        steps = [("preprocessor", pre), ("selector", sel), ("classifier", hgb)]
    alt = Pipeline(steps)

fitted = mlca_runtime.fit_record(alt, X, "train", y=y)
mlca_runtime.save_model(fitted, "alternative_model.pkl")

print(f"train rows: {len(X)}")
print(f"features: {len(cols)}")
print(f"classes: {list(getattr(fitted, 'classes_', []))}")
