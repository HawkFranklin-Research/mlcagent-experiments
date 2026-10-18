import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
targets = proto["targets"]
excluded = proto.get("excluded_columns", []) or []
group_col = proto.get("group_column")
task = proto["task"]

train = pd.read_parquet(inp["train"])
pre = pickle.loads(inp["05_preprocess_fit_preprocessing_0FJ6AMWD/preprocessor.pkl"].read_bytes())
sel = pickle.loads(inp["06_features_select_features_79Q93RHQ/selector.pkl"].read_bytes())

if isinstance(sel, Pipeline) and len(sel.named_steps) > 1:
    frontend, steps = sel, list(sel.steps)
    feat_src = sel
elif isinstance(pre, Pipeline):
    frontend = pre
    steps = list(pre.steps) + [("selector", sel)]
    feat_src = pre
else:
    frontend = Pipeline([("preprocessing", pre), ("selector", sel)])
    steps = [("preprocessing", pre), ("selector", sel)]
    feat_src = pre

drop = set(targets) | set(excluded) | ({group_col} if group_col else set())

def keylike(s):
    try:
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            return s.nunique(dropna=False) == len(s)
    except Exception:
        return False
    return False

if hasattr(feat_src, "feature_names_in_"):
    feat_cols = [c for c in feat_src.feature_names_in_ if c in train.columns]
else:
    feat_cols = [c for c in train.columns if c not in drop]
feat_cols = [c for c in feat_cols if c not in drop and not keylike(train[c])]

X_train = train[feat_cols]
y_train = train[targets[0]]

steps.append(("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)))
model = Pipeline(steps)

mlca_runtime.fit_record(model, X_train, "train", y=y_train)
mlca_runtime.save_model(model, "model.pkl")

print(f"n_train={len(train)} n_features={len(feat_cols)} n_steps={len(model.steps)}")
print(f"steps={[s[0] for s in model.steps]}")
print(f"classes={list(model.classes_)}")
