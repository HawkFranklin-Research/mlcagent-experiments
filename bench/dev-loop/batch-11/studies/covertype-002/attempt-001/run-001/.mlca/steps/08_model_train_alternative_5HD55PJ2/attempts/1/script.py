import json
import pickle

import pandas as pd
import mlca_runtime
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

SEED = 42
mlca_runtime.set_seed(SEED)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column", None)
_ = protocol["task"]["primary_metric"]

train = pd.read_parquet(inputs["train"])
if len(targets) != 1:
    raise ValueError(f"expected one target, got {targets}")
target = targets[0]
y = train[target]


def fallback_features(df):
    cols = []
    for c in df.columns:
        if c in targets or c in excluded or c == group:
            continue
        s = df[c]
        if s.nunique(dropna=False) == len(df):
            if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                continue
            if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
                continue
        cols.append(c)
    return cols


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


model = load(inputs["07_model_train_logistic_H5RFG7PA/model.pkl"])

if isinstance(model, Pipeline):
    raw_cols = (
        list(model.feature_names_in_)
        if hasattr(model, "feature_names_in_")
        else fallback_features(train)
    )
    pipe = clone(model)
    last_name = pipe.steps[-1][0]
    pipe.steps[-1] = (
        last_name,
        HistGradientBoostingClassifier(random_state=SEED),
    )
else:
    pre = load(inputs["05_preprocess_fit_preprocessing_8APC2384/preprocessor.pkl"])
    sel = load(inputs["06_features_select_features_23ZRTGDS/selector.pkl"])
    raw_cols = (
        list(pre.feature_names_in_)
        if hasattr(pre, "feature_names_in_")
        else fallback_features(train)
    )
    if isinstance(sel, Pipeline) and hasattr(sel, "feature_names_in_"):
        front = clone(sel)
        pipe = Pipeline(
            front.steps
            + [("model", HistGradientBoostingClassifier(random_state=SEED))]
        )
    else:
        pipe = Pipeline(
            [
                ("pre", clone(pre)),
                ("sel", clone(sel)),
                ("model", HistGradientBoostingClassifier(random_state=SEED)),
            ]
        )

bad = set(raw_cols) & (
    set(targets) | set(excluded) | ({group} if group else set())
)
if bad:
    raise ValueError(f"target/excluded/group columns in features: {sorted(bad)}")

missing = set(raw_cols) - set(train.columns)
if missing:
    raise ValueError(f"missing feature columns: {sorted(missing)}")

X = train[raw_cols]
fitted = mlca_runtime.fit_record(pipe, X, "train", y=y)
out = mlca_runtime.save_model(fitted, "alternative_model.pkl")

print(f"rows={len(X)} features={len(raw_cols)}")
print(f"saved={out.name}")
