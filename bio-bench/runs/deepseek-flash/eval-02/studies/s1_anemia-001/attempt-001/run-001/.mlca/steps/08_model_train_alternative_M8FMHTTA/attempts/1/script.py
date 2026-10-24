import json
import pickle

import pandas as pd

import mlca_runtime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

seed = 42
mlca_runtime.set_seed(seed)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task", "group_column"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available: {list(protocol)}")

targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"] or [])
group_col = protocol["group_column"]
train = pd.read_parquet(inputs["train"])

def key_like(s):
    if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        name = s.name.lower()
        return s.nunique(dropna=True) == len(s) and any(
            t in name for t in ("id", "code", "key", "index", "row")
        )
    return False

base_feature_cols = [
    c
    for c in train.columns
    if c not in targets
    and c not in excluded
    and c != group_col
    and not key_like(train[c])
]

with open(inputs["07_model_train_logistic_20PTX4M5/model.pkl"], "rb") as f:
    model = pickle.load(f)

if hasattr(model, "feature_names_in_"):
    expected = list(model.feature_names_in_)
    bad = set(expected) & (set(targets) | excluded | ({group_col} if group_col else set()))
    if bad:
        raise ValueError(f"Saved model expects excluded columns: {bad}")
    missing = [c for c in expected if c not in train.columns]
    if missing:
        raise KeyError(f"Saved model expects missing columns: {missing}")
    feature_cols = expected
else:
    feature_cols = base_feature_cols

X = train[feature_cols].copy()
y = train[targets[0]].copy()

hgb = HistGradientBoostingClassifier(random_state=seed)
if "class_weight" in hgb.get_params():
    counts = y.value_counts(normalize=True)
    if len(counts) >= 2 and counts.min() < 0.1:
        hgb.set_params(class_weight="balanced")

if isinstance(model, Pipeline):
    last_name = model.steps[-1][0]
    model.set_params(**{last_name: hgb})
    pipeline = model
else:
    with open(inputs["05_preprocess_fit_preprocessing_HCSJFKGG/preprocessor.pkl"], "rb") as f:
        pre = pickle.load(f)
    with open(inputs["06_features_select_features_5E6WDF2E/selector.pkl"], "rb") as f:
        sel = pickle.load(f)
    sel_steps = sel.named_steps if isinstance(sel, Pipeline) else {}
    if isinstance(sel, Pipeline) and any(
        n in ("preprocessor", "preprocess", "preprocessing") for n in sel_steps
    ):
        pipeline = Pipeline([("selector", sel), ("model", hgb)])
    else:
        pipeline = Pipeline([("preprocessor", pre), ("selector", sel), ("model", hgb)])

mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(pipeline, "alternative_model.pkl")

print(f"train rows={len(train)}, features={len(feature_cols)}")
print(f"saved alternative_model.pkl with {type(hgb).__name__}")
