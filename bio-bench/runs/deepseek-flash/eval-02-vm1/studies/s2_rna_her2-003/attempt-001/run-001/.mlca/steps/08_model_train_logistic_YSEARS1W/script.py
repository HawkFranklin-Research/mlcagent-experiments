import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load(inputs["06_preprocess_fit_preprocessing_VD4Z4Z0N/preprocessor.pkl"])
sel = load(inputs["07_features_select_features_AKBM21YH/selector.pkl"])

def front_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, s in obj.steps:
            if hasattr(s, "feature_names_in_"):
                return list(s.feature_names_in_)
    return None

pre_cols = front_cols(pre)
sel_cols = front_cols(sel)

if pre_cols and sel_cols and pre_cols == sel_cols:
    steps = [("sel", sel)]
else:
    steps = [("pre", pre), ("sel", sel)]
steps.append(("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)))
pipe = Pipeline(steps)

drop = set(targets) | excluded
if group_col:
    drop.add(group_col)
feat_cols = [c for c in (pre_cols or sel_cols or [c for c in train.columns if c not in drop])
             if c in train.columns]

X = train[feat_cols]
y = train[targets[0]]
mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(pipe, "model.pkl")

vc = pd.Series(y).value_counts().to_dict()
print(f"rows={len(train)} feats={len(feat_cols)} labels={vc}")
