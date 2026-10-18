import json
import pickle

import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())


def required(mapping, key):
    if key not in mapping:
        raise KeyError(f"Missing {key!r}; available top-level keys: {sorted(mapping)}")
    return mapping[key]


targets = required(protocol, "targets")
excluded = required(protocol, "excluded_columns")
group_col = required(protocol, "group_column")

train = pd.read_parquet(inputs["train"])
for target in targets:
    if target not in train.columns:
        raise KeyError(f"Target {target!r} not found in training table")


def is_keylike(series, name):
    if name in targets or name in excluded or name == group_col:
        return True
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        if len(series) and series.nunique(dropna=False) == len(series):
            return True
    if pd.api.types.is_integer_dtype(series) and not pd.api.types.is_bool_dtype(series):
        lowered = name.lower()
        if (
            len(series)
            and series.nunique(dropna=False) == len(series)
            and any(token in lowered for token in ("id", "key", "code", "record"))
        ):
            return True
    return False


feature_cols = [col for col in train.columns if not is_keylike(train[col], col)]
if not feature_cols:
    raise ValueError("No feature columns remain")
if set(feature_cols) & set(targets):
    raise AssertionError("Target leaked into feature columns")

with open(inputs["06_preprocess_fit_preprocessing_DB5T0R19/preprocessor.pkl"], "rb") as handle:
    preprocessor = pickle.load(handle)
with open(inputs["07_features_retain_features_05E7EJVW/selector_all.pkl"], "rb") as handle:
    selector = pickle.load(handle)


def includes_preprocessing(obj):
    if isinstance(obj, ColumnTransformer):
        return True
    if isinstance(obj, Pipeline):
        for step_name, step in obj.named_steps.items():
            if step_name.lower() in ("preprocessor", "preprocess", "preprocessing"):
                return True
            if includes_preprocessing(step):
                return True
    return False


selector_has_preprocessing = includes_preprocessing(selector)
frontend = selector if selector_has_preprocessing else preprocessor
if hasattr(frontend, "feature_names_in_"):
    expected = list(frontend.feature_names_in_)
    missing = [col for col in expected if col not in train.columns]
    if missing:
        raise KeyError(f"Frontend expects missing columns: {missing}")
    feature_cols = expected
if set(feature_cols) & set(targets):
    raise AssertionError("Target leaked into frontend input columns")

X_train = train[feature_cols].copy()
y_train = train[targets].copy()

model = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=100, random_state=42)
)
if selector_has_preprocessing:
    pipeline = Pipeline([("frontend", selector), ("model", model)])
else:
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("selector", selector),
            ("model", model),
        ]
    )

pipeline = mlca_runtime.fit_record(pipeline, X_train, "train", y_train)
mlca_runtime.save_model(pipeline, "alternative_model.pkl")

print(f"rows={len(train)} features={len(feature_cols)} targets={len(targets)}")
print(f"frontend={'selector' if selector_has_preprocessing else 'preprocessor+selector'}")
print("saved=alternative_model.pkl")
