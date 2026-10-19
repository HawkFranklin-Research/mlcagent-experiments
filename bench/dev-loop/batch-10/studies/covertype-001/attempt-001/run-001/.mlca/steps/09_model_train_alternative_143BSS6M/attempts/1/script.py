import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

try:
    targets = protocol["targets"]
    excluded = protocol.get("excluded_columns", [])
    group_col = protocol.get("group_column")
    task = protocol["task"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available top-level keys: {list(protocol)}")

train = pd.read_parquet(inputs["train"])
_ = pd.read_parquet(inputs["val"])

with open(inputs["08_model_train_logistic_6YJ0NC7Y/model.pkl"], "rb") as f:
    base = pickle.load(f)

if isinstance(base, Pipeline) and base.steps:
    steps = list(base.steps)
    steps[-1] = ("model", HistGradientBoostingClassifier(random_state=42))
    pipe = Pipeline(steps)
else:
    with open(inputs["06_preprocess_fit_preprocessing_XZ1WZNJH/preprocessor.pkl"], "rb") as f:
        pre = pickle.load(f)
    with open(inputs["07_features_select_features_KRQ7615Z/selector.pkl"], "rb") as f:
        sel = pickle.load(f)
    pipe = Pipeline([
        ("preprocessor", pre),
        ("selector", sel),
        ("model", HistGradientBoostingClassifier(random_state=42)),
    ])

feature_cols = list(getattr(pipe, "feature_names_in_", []))
if not feature_cols:
    feature_cols = list(getattr(pipe.steps[0][1], "feature_names_in_", []))
if not feature_cols:
    forbidden = set(targets) | set(excluded) | ({group_col} if group_col else set())
    feature_cols = []
    for c in train.columns:
        if c in forbidden:
            continue
        s = train[c]
        if s.nunique(dropna=True) == len(train) and (
            pd.api.types.is_object_dtype(s) or pd.api.types.is_integer_dtype(s)
        ):
            continue
        feature_cols.append(c)

forbidden = set(targets) | set(excluded) | ({group_col} if group_col else set())
bad = forbidden & set(feature_cols)
if bad:
    raise ValueError(f"Feature list contains forbidden columns: {bad}")

X = train[feature_cols].copy()
y = train[targets[0]]

mlca_runtime.fit_record(pipe, X, partition="train", y=y)
mlca_runtime.save_model(pipe, "alternative_model.pkl")
print("train", train.shape, "features", len(feature_cols))
print("saved alternative_model.pkl")
