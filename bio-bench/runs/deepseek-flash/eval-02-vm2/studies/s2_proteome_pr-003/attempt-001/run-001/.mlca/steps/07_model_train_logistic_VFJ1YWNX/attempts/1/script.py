import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = proto["targets"]
task = proto["task"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

with open(inputs["05_preprocess_fit_preprocessing_YCEBPQHT/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_FXHKDZPP/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def steps_of(obj):
    return list(obj.steps) if isinstance(obj, Pipeline) else [("est", obj)]

def first_names(obj):
    if isinstance(obj, Pipeline):
        return first_names(obj.steps[0][1])
    return getattr(obj, "feature_names_in_", None)

pre_in = first_names(pre)
sel_in = first_names(sel)

if pre_in is not None:
    feat_cols = list(pre_in)
elif sel_in is not None:
    feat_cols = list(sel_in)
else:
    excl = set(targets) | set(proto["excluded_columns"])
    if proto.get("group_column"):
        excl.add(proto["group_column"])
    feat_cols = [c for c in train.columns if c not in excl]

lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

includes_pre = (sel_in is not None and pre_in is not None and list(sel_in) == list(pre_in))
if includes_pre:
    combined = [(f"s{i}_{n}", o) for i, (n, o) in enumerate(steps_of(sel))]
else:
    combined = [(f"p{i}_{n}", o) for i, (n, o) in enumerate(steps_of(pre))] + \
               [(f"s{i}_{n}", o) for i, (n, o) in enumerate(steps_of(sel))]
pipe = Pipeline(combined + [("clf", lr)])

y = train[targets[0]]
X = train[feat_cols]
print("train", X.shape, "features", len(feat_cols), "selector_includes_pre", includes_pre)
mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(pipe, "model.pkl")
print("saved model.pkl")
