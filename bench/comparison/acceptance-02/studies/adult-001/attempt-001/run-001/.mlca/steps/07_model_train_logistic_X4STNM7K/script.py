import json, pickle
from pathlib import Path
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(Path(inputs["_protocol"]).read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing protocol key {k!r}; available top-level keys: {list(d.keys())}")
    return d[k]

targets = list(need(protocol, "targets"))
excluded = set(need(protocol, "excluded_columns"))
group = protocol.get("group_column")
need(protocol, "task")
target = targets[0]

train = pd.read_parquet(inputs["train"])

def base_features(df):
    cols = []
    n = len(df)
    for c in df.columns:
        if c in targets or c in excluded or (group is not None and c == group):
            continue
        s = df[c]
        try:
            nu = s.nunique(dropna=False)
        except Exception:
            nu = n + 1
        if nu == n:
            lname = str(c).lower()
            is_key_name = any(t in lname for t in ("id", "key", "index", "row"))
            is_text = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
            if is_key_name or is_text:
                continue
        cols.append(c)
    return cols

base_cols = base_features(train)

with open(inputs["05_preprocess_fit_preprocessing_ABA216ED/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_select_features_DJ7CCJDF/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

def input_features(est):
    if hasattr(est, "feature_names_in_"):
        try:
            return list(est.feature_names_in_)
        except Exception:
            pass
    if isinstance(est, Pipeline):
        steps = list(est.named_steps.values())
        if steps and hasattr(steps[0], "feature_names_in_"):
            try:
                return list(steps[0].feature_names_in_)
            except Exception:
                pass
    return None

def output_features(est):
    if hasattr(est, "get_feature_names_out"):
        try:
            return list(est.get_feature_names_out())
        except Exception:
            return None
    if isinstance(est, Pipeline):
        steps = list(est.named_steps.values())
        if steps and hasattr(steps[-1], "get_feature_names_out"):
            try:
                return list(steps[-1].get_feature_names_out())
            except Exception:
                return None
    return None

def has_preprocessing(est):
    if isinstance(est, ColumnTransformer):
        return True
    if isinstance(est, Pipeline):
        for step in est.named_steps.values():
            if has_preprocessing(step):
                return True
    return False

if has_preprocessing(selector):
    frontend = selector
    final_pipeline = Pipeline([("selector", selector), ("clf", None)])
else:
    sel_in = input_features(selector)
    pp_out = output_features(preprocessor)
    if sel_in is not None and pp_out is not None and set(sel_in) != set(pp_out) and set(sel_in) == set(base_cols):
        frontend = selector
        final_pipeline = Pipeline([("selector", selector), ("clf", None)])
    else:
        frontend = preprocessor
        final_pipeline = Pipeline([("preprocessor", preprocessor), ("selector", selector), ("clf", None)])

front_in = input_features(frontend)
if front_in is not None:
    forbidden = set(targets) | excluded | ({group} if group is not None else set())
    bad = forbidden & set(front_in)
    if bad:
        raise ValueError(f"Frontend expects forbidden columns: {bad}")
    missing = [c for c in front_in if c not in train.columns]
    if missing:
        raise ValueError(f"Frontend expects missing columns: {missing}")
    feature_cols = list(front_in)
else:
    feature_cols = base_cols

X_train = train[feature_cols].copy()
y_train = train[target]

if len(train) > 100000:
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="saga")
else:
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
final_pipeline.set_params(clf=clf)

fitted = mlca_runtime.fit_record(final_pipeline, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "model.pkl")
print(f"train rows={len(train)}, features={len(feature_cols)}")
print(f"steps={list(fitted.named_steps.keys())}")
print("saved model.pkl")
