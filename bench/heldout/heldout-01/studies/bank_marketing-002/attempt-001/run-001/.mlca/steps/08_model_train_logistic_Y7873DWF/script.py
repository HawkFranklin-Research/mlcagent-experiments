import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
train = pd.read_parquet(inp["train"])


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


pre = load_pkl(inp["06_preprocess_fit_preprocessing_STWKFGVD/preprocessor.pkl"])
sel = load_pkl(inp["07_features_select_features_N6F5BDYX/selector.pkl"])


def first_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            names = first_names(step)
            if names:
                return names
    return None


def has_pre(obj):
    if not isinstance(obj, Pipeline):
        return False
    for name, step in obj.steps:
        if isinstance(step, ColumnTransformer):
            return True
        if name.lower() in {"preprocessor", "preprocess", "preprocessing"}:
            return True
        if has_pre(step):
            return True
    return False


pre_cols = first_names(pre)
sel_cols = first_names(sel)
sel_wrapped = has_pre(sel)
if not sel_wrapped and isinstance(sel, Pipeline) and pre_cols and sel_cols and pre_cols == sel_cols and len(sel.steps) > 1:
    sel_wrapped = True

frontend = sel if sel_wrapped else pre
cols = first_names(frontend)
if not cols:
    drop = set(targets) | set(excluded) | ({group} if group else set())
    cols = [c for c in train.columns if c not in drop]

for c in cols:
    if c in targets or c in excluded or c == group:
        raise ValueError(f"non-feature column reached model: {c}")
missing = [c for c in (first_names(frontend) or []) if c not in train.columns]
if missing:
    raise KeyError(f"missing expected feature columns: {missing}")

steps = [("preprocessing", frontend)]
if not sel_wrapped:
    steps.append(("selector", sel))
steps.append(("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)))
model = Pipeline(steps)

X = train[cols].copy()
y = train[targets[0]]
model = mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(model, "model.pkl")
print(f"train rows: {len(train)}; features: {len(cols)}")
print(f"selector wrapped preprocessing: {sel_wrapped}")
print("saved model.pkl")
