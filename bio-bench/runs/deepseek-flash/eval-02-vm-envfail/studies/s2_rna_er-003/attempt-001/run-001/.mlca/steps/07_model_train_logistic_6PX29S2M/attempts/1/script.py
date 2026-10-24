import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
_ = pd.read_parquet(inputs["val"])

with open(inputs["05_preprocess_fit_preprocessing_K9GV8476/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_BXHGCTQP/selector.pkl"], "rb") as f:
    sel = pickle.load(f)


def first_inputs(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline) and obj.steps:
        inner = obj.steps[0][1]
        if hasattr(inner, "feature_names_in_"):
            return list(inner.feature_names_in_)
    return None


def has_pre_step(obj):
    return isinstance(obj, Pipeline) and any(
        k in obj.named_steps
        for k in ("preprocessor", "preprocessing", "preprocess", "pre", "prep")
    )


sel_in = first_inputs(sel)
pre_in = first_inputs(pre)
pre_out = None
try:
    pre_out = [str(c) for c in pre.get_feature_names_out()]
except Exception:
    pre_out = None

if has_pre_step(sel):
    stages = [("selector", sel)]
elif pre_out is not None and sel_in is not None and list(sel_in) == pre_out:
    stages = [("pre", pre), ("selector", sel)]
elif pre_in is not None and sel_in is not None and list(pre_in) == list(sel_in):
    stages = [("selector", sel)]
else:
    stages = [("pre", pre), ("selector", sel)]

stages.append(("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)))
full = Pipeline(stages)

X_cols = first_inputs(stages[0][1])
if X_cols is None:
    drop = set(targets) | excluded | ({group_col} if group_col else set())
    X_cols = [c for c in train.columns if c not in drop]

X = train[X_cols]
y = train[targets[0]]

mlca_runtime.fit_record(full, X, partition="train", y=y)
mlca_runtime.save_model(full, "model.pkl")

print(f"steps={[s[0] for s in full.steps]}")
print(f"n_features={len(X_cols)} n_rows={len(train)}")
