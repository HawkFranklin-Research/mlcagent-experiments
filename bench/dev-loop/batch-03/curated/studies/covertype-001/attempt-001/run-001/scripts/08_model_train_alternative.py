import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

proto = json.loads(inputs["_protocol"].read_text())
targets = proto["targets"]
excluded = proto.get("excluded_columns", []) or []
group = proto.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_keylike(s):
    if s.dtype == object:
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == len(s)
    return False

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feat_cols = [c for c in train.columns if c not in drop and not is_keylike(train[c])]

with open(inputs["05_preprocess_fit_preprocessing_1A6AJYGR/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_JT24GMYZ/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def frontend_cols(obj):
    if isinstance(obj, Pipeline):
        return frontend_cols(next(iter(obj.named_steps.values())))
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    return None

sel_first = next(iter(sel.named_steps.values())) if isinstance(sel, Pipeline) else sel
sel_has_pre = isinstance(sel_first, ColumnTransformer)

if sel_has_pre:
    steps = [("selector", sel)]
    frontend = sel
else:
    steps = [("preprocessor", pre), ("selector", sel)]
    frontend = pre

cols = frontend_cols(frontend) or feat_cols
X = train[cols]
y = train[targets[0]]

clf = HistGradientBoostingClassifier(random_state=42)
pipe = Pipeline(steps + [("model", clf)])

mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(pipe, "alternative_model.pkl")

print("features:", len(cols))
print("train:", X.shape, "val:", val.shape)
print("classes:", list(pipe.named_steps["model"].classes_))
