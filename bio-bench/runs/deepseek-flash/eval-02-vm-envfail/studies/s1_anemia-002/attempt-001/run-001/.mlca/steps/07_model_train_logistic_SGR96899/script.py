import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
task = protocol["task"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

with open(inputs["05_preprocess_fit_preprocessing_3WJD77DF/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_retain_features_6D2RD7T4/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)


def input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None


pre_in = input_cols(preprocessor)
if pre_in is not None:
    feature_cols = [c for c in pre_in if c in train.columns]
else:
    drop = set(targets) | set(excluded)
    if group_col:
        drop.add(group_col)
    cand = [c for c in train.columns if c not in drop]
    n = len(train)
    feature_cols = []
    for c in cand:
        s = train[c]
        nu = s.nunique(dropna=True)
        if nu == n:
            if s.dtype == object or pd.api.types.is_string_dtype(s):
                continue
            if pd.api.types.is_integer_dtype(s):
                sv = s.dropna().sort_values().to_numpy()
                if len(sv) > 1 and (sv[1:] - sv[:-1] == 1).all():
                    continue
        feature_cols.append(c)

sel_in = input_cols(selector)
clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

if sel_in is not None and set(sel_in) == set(feature_cols):
    steps = list(selector.steps) if isinstance(selector, Pipeline) else [("sel", selector)]
    pipeline = Pipeline(steps + [("clf", clf)])
else:
    pipeline = Pipeline([("pre", preprocessor), ("sel", selector), ("clf", clf)])

X = train[feature_cols]
y = train[targets[0]]
fitted = mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(fitted, "model.pkl")

if task["primary_metric"] == "roc_auc":
    classes = fitted.named_steps["clf"].classes_
    proba = fitted.predict_proba(val[feature_cols])
    if len(classes) == 2:
        y_true = (val[targets[0]] == classes[1]).astype(int)
        mlca_runtime.record_metric("roc_auc", float(roc_auc_score(y_true, proba[:, 1])), "val")

print(f"train={train.shape} val={val.shape} n_features={len(feature_cols)}")
