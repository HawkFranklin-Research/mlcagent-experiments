import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing {key}; available keys: {list(protocol)}")
targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group = protocol["group_column"]

train = pd.read_parquet(paths["train"])
with open(paths["05_preprocess_fit_preprocessing_XJYNH9KV/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(paths["06_features_select_features_61DSRHRD/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_cols(est):
    if est is None:
        return None
    if hasattr(est, "feature_names_in_"):
        return list(est.feature_names_in_)
    if hasattr(est, "steps"):
        for _, step in est.steps:
            cols = input_cols(step)
            if cols is not None:
                return cols
    return None

pre_cols = input_cols(pre)
sel_cols = input_cols(sel)

if pre_cols is None and sel_cols is None:
    cols = [c for c in train.columns if c not in set(targets) | excluded and c != group]
    cols = [c for c in cols if not (train[c].dtype == object and train[c].nunique() == len(train))]
elif pre_cols is not None and sel_cols is not None and set(pre_cols) == set(sel_cols):
    cols = sel_cols
else:
    cols = pre_cols if pre_cols is not None else sel_cols

if sel is not None and pre is not None and pre_cols is not None and sel_cols is not None:
    if set(sel_cols) == set(pre_cols):
        frontend = sel
        cols = sel_cols
    else:
        try:
            pre_out = pre.get_feature_names_out()
        except Exception:
            pre_out = None
        if pre_out is not None and set(sel_cols).issubset(set(pre_out)):
            frontend = Pipeline([("preprocessor", pre), ("selector", sel)])
            cols = pre_cols
        else:
            frontend = sel if sel_cols is not None else Pipeline([("preprocessor", pre), ("selector", sel)])
            cols = sel_cols if sel_cols is not None else pre_cols
elif sel is not None:
    frontend = sel
    cols = sel_cols or cols
else:
    frontend = pre
    cols = pre_cols or cols

cols = [c for c in cols if c not in set(targets) | excluded and c != group]
model = Pipeline([
    ("frontend", frontend),
    ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
])
X = train[cols]
y = train[targets[0]] if len(targets) == 1 else train[targets]
mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(model, "model.pkl")
print(f"train={train.shape} features={len(cols)} model saved")
