import json
import pickle

import mlca_runtime
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

seed = 42
mlca_runtime.set_seed(seed)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])

targets = protocol["targets"]
target = targets[0]
y = train[target]

base = pickle.loads(inputs["07_model_train_logistic_HJ9ARHSV/model.pkl"].read_bytes())
excluded = set(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
bad = set(targets) | excluded | ({group} if group else set())

raw_cols = [c for c in train.columns if c not in bad]
drop_key = []
for c in raw_cols:
    s = train[c]
    if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
        if s.nunique(dropna=False) == len(s):
            drop_key.append(c)
    elif pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=False) == len(s):
        drop_key.append(c)
raw_cols = [c for c in raw_cols if c not in drop_key]

hgb = HistGradientBoostingClassifier(random_state=seed)

if isinstance(base, Pipeline):
    steps = list(base.steps)
    last_name = steps[-1][0]
    steps[-1] = (last_name, hgb)
    pipe = Pipeline(steps)
    cols = list(getattr(base, "feature_names_in_", []))
    if cols and all(c in train.columns for c in cols):
        if set(cols) & bad:
            raise ValueError("Saved pipeline expects target or excluded columns")
        feature_cols = cols
    else:
        feature_cols = raw_cols
        pre = pickle.loads(inputs["05_preprocess_fit_preprocessing_Y84CNP1M/preprocessor.pkl"].read_bytes())
        sel = pickle.loads(inputs["06_features_select_features_TG41SQ76/selector.pkl"].read_bytes())
        if isinstance(sel, Pipeline) and getattr(sel, "feature_names_in_", None) is not None and all(
            c in train.columns for c in sel.feature_names_in_
        ):
            pipe = Pipeline([("selector", sel), ("classifier", hgb)])
            feature_cols = list(sel.feature_names_in_)
        else:
            pipe = Pipeline([("preprocessor", pre), ("selector", sel), ("classifier", hgb)])
else:
    feature_cols = raw_cols
    pre = pickle.loads(inputs["05_preprocess_fit_preprocessing_Y84CNP1M/preprocessor.pkl"].read_bytes())
    sel = pickle.loads(inputs["06_features_select_features_TG41SQ76/selector.pkl"].read_bytes())
    if isinstance(sel, Pipeline) and getattr(sel, "feature_names_in_", None) is not None and all(
        c in train.columns for c in sel.feature_names_in_
    ):
        pipe = Pipeline([("selector", sel), ("classifier", hgb)])
        feature_cols = list(sel.feature_names_in_)
    else:
        pipe = Pipeline([("preprocessor", pre), ("selector", sel), ("classifier", hgb)])

X = train[feature_cols]
fitted = mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(fitted if fitted is not None else pipe, "alternative_model.pkl")
print("features", len(feature_cols), "train", X.shape)
print("saved alternative_model.pkl")
