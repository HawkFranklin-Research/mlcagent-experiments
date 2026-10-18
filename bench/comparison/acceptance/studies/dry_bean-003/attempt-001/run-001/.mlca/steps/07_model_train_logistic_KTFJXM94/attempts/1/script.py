import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
train = pd.read_parquet(inp["train"])

targets = protocol["targets"]
excluded = list(protocol.get("excluded_columns") or [])
group = protocol.get("group_column")
drop = set(targets) | set(excluded) | ({group} if group else set())
y = train[targets[0]]
X = train[[c for c in train.columns if c not in drop]]

with open(inp["05_preprocess_fit_preprocessing_DYJSGEW8/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_select_features_KAD36QCW/selector.pkl"], "rb") as f:
    sel = pickle.load(f)


def _contains(pipe, obj):
    if not isinstance(pipe, Pipeline):
        return False
    return any(s is obj or s == obj for s in pipe.named_steps.values())


if _contains(sel, pre):
    frontend = sel
elif _contains(pre, sel):
    frontend = pre
else:
    frontend = Pipeline([("pre", pre), ("sel", sel)])


def _input_names(frontend):
    names = getattr(frontend, "feature_names_in_", None)
    if names is not None:
        return list(names)
    if isinstance(frontend, Pipeline):
        for step in frontend.named_steps.values():
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                return list(names)
    return None


names = _input_names(frontend)
if names:
    missing = [c for c in names if c not in X.columns]
    if missing:
        raise KeyError(f"missing expected input columns: {missing}")
    X = X[names]

model = Pipeline([
    ("frontend", frontend),
    ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
])

mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(model, "model.pkl")
mlca_runtime.record_fact_candidate("n_train_rows", int(len(X)), source="model.pkl")
print("saved model.pkl", "n_train", len(X), "n_features", X.shape[1])
