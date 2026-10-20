import json
import pickle

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

import mlca_runtime

seed = 42
mlca_runtime.set_seed(seed)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])

def need(key):
    if key not in protocol:
        raise KeyError(f"{key} missing; available top-level keys: {list(protocol.keys())}")
    return protocol[key]

targets = need("targets")
excluded = need("excluded_columns")
group = need("group_column")
target = targets[0]
y = train[target]
forbidden = set(targets) | set(excluded)
if group is not None:
    forbidden.add(group)

def derive_features(df, targets, excluded, group):
    forbidden = set(targets) | set(excluded)
    if group is not None:
        forbidden.add(group)
    n = len(df)
    out = []
    for c in df.columns:
        if c in forbidden:
            continue
        s = df[c]
        is_text = (
            pd.api.types.is_object_dtype(s.dtype)
            or pd.api.types.is_string_dtype(s.dtype)
            or isinstance(s.dtype, pd.CategoricalDtype)
        )
        if is_text and s.nunique(dropna=False) == n:
            continue
        if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=False) == n:
            lc = str(c).lower()
            if lc.endswith("id") or lc.startswith("id") or "_id" in lc:
                continue
        out.append(c)
    return out

with open(inputs["07_model_train_logistic_QHAMT8ZK/model.pkl"], "rb") as f:
    base = pickle.load(f)

if isinstance(base, Pipeline) and len(base.steps) >= 2 and hasattr(base, "feature_names_in_"):
    feature_cols = list(base.feature_names_in_)
    if forbidden & set(feature_cols):
        raise ValueError("Saved pipeline expects forbidden target/excluded/group columns")
    last_name = base.steps[-1][0]
    clf = HistGradientBoostingClassifier(random_state=seed)
    pipe = Pipeline(list(base.steps[:-1]) + [(last_name, clf)])
else:
    with open(inputs["05_preprocess_fit_preprocessing_XHNDRHCS/preprocessor.pkl"], "rb") as f:
        pre = pickle.load(f)
    with open(inputs["06_features_select_features_4TDFQMJC/selector.pkl"], "rb") as f:
        sel = pickle.load(f)

    if hasattr(pre, "feature_names_in_"):
        feature_cols = list(pre.feature_names_in_)
    else:
        feature_cols = derive_features(train, targets, excluded, group)
    if forbidden & set(feature_cols):
        raise ValueError("Feature list contains forbidden target/excluded/group columns")

    clf = HistGradientBoostingClassifier(random_state=seed)
    sel_raw = (
        isinstance(sel, Pipeline)
        and hasattr(sel, "named_steps")
        and hasattr(sel, "feature_names_in_")
        and set(sel.feature_names_in_) == set(feature_cols)
    )
    if sel_raw:
        pipe = Pipeline([("selector", sel), ("classifier", clf)])
    else:
        pipe = Pipeline([
            ("preprocessor", pre),
            ("selector", sel),
            ("classifier", clf),
        ])

X = train.loc[:, feature_cols]
mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(pipe, "alternative_model.pkl")
print("train", train.shape)
print("features", len(feature_cols))
print("saved alternative_model.pkl")
