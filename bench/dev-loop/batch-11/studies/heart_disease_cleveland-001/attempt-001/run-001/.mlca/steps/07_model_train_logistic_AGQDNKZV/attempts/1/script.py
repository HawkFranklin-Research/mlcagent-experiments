import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])

with open(inputs["05_preprocess_fit_preprocessing_69FZY9R2/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_TEGHQMDA/selector.pkl"], "rb") as f:
    sel = pickle.load(f)


def first_in(obj):
    if obj is None:
        return None
    cols = getattr(obj, "feature_names_in_", None)
    if cols is not None:
        return list(cols)
    if isinstance(obj, Pipeline) and obj.steps:
        return first_in(obj.steps[0][1])
    return None


def out_cols(obj):
    try:
        return list(obj.get_feature_names_out())
    except Exception:
        return None


pre_in = first_in(pre)
sel_in = first_in(sel)
pre_out = out_cols(pre)

if pre_in is None:
    n = len(train)
    pre_in = [c for c in train.columns
              if c not in targets and c not in excluded and c != group
              and not (pd.api.types.is_integer_dtype(train[c])
                       and train[c].nunique() == n and n > 20)
              and not (train[c].dtype == object
                       and train[c].nunique() == n and n > 20)]

if sel_in is not None and list(sel_in) == list(pre_in):
    frontend = sel
    feat_cols = list(pre_in)
elif sel_in is not None and pre_out is not None and list(sel_in) == list(pre_out):
    frontend = Pipeline([("pre", pre), ("sel", sel)])
    feat_cols = list(pre_in)
else:
    frontend = Pipeline([("pre", pre), ("sel", sel)])
    feat_cols = list(pre_in)

pipe = Pipeline([
    ("front", frontend),
    ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
])

X = train[feat_cols]
y = train[targets[0]]

mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(pipe, "model.pkl")
print(f"rows={len(train)} features={len(feat_cols)} pos_rate={float(y.mean()):.4f}")
