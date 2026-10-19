import json, pickle
import mlca_runtime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
targets = proto["targets"]
excluded = proto.get("excluded_columns", []) or []
group = proto.get("group_column")
target = targets[0]

train = pd.read_parquet(inp["train"])
pre = pickle.loads(inp["05_preprocess_fit_preprocessing_H4RCZJ73/preprocessor.pkl"].read_bytes())
sel = pickle.loads(inp["06_features_select_features_F6NPFNPS/selector.pkl"].read_bytes())

def in_cols(o):
    if hasattr(o, "feature_names_in_"):
        return [str(c) for c in o.feature_names_in_]
    if isinstance(o, Pipeline):
        for s in o.named_steps.values():
            if hasattr(s, "feature_names_in_"):
                return [str(c) for c in s.feature_names_in_]
    return None

raw = in_cols(pre) or list(train.columns)
feat_cols = [c for c in raw if c not in targets and c not in excluded and c != group and c in train.columns]
X = train[feat_cols].copy()
y = train[target]
mask = y.notna()
X, y = X.loc[mask], y.loc[mask]

pre_out = list(pre.get_feature_names_out()) if hasattr(pre, "get_feature_names_out") else None
sel_in = in_cols(sel)

clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
if sel_in is not None and set(sel_in) == set(feat_cols):
    X = X[sel_in]
    pipe = Pipeline([("sel", sel), ("clf", clf)])
elif sel_in is not None and pre_out is not None and set(sel_in) == set(pre_out):
    pipe = Pipeline([("pre", pre), ("sel", sel), ("clf", clf)])
else:
    pipe = Pipeline([("pre", pre), ("sel", sel), ("clf", clf)])

pipe = mlca_runtime.fit_record(pipe, X, partition="train", y=y)
mlca_runtime.save_model(pipe, "model.pkl")
print("features", len(feat_cols), "rows", len(X), "classes", len(pipe.classes_))
print("saved model.pkl")
